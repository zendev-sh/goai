package goai

import (
	"errors"
	"runtime/debug"
)

// firePanicHooks notifies every registered OnPanic observer. It never itself
// panics: a panic inside an OnPanic callback is recovered and discarded so it
// cannot disrupt the panic-handling flow it is observing.
func firePanicHooks(onPanic []func(PanicInfo), phase string, r any, stack []byte) {
	for _, fn := range onPanic {
		func(f func(PanicInfo)) {
			defer func() { _ = recover() }()
			f(PanicInfo{Phase: phase, Value: r, Stack: stack})
		}(fn)
	}
}

// firePanic captures the current stack and notifies the OnPanic observers. It
// is used by the resilient tool path, which recovers and continues (converting
// the panic to a tool error) rather than propagating it.
func firePanic(onPanic []func(PanicInfo), phase string, r any) {
	firePanicHooks(onPanic, phase, r, debug.Stack())
}

// newPanicError captures the current stack, fires the OnPanic observers exactly
// once, and returns the resulting *PanicError.
func newPanicError(onPanic []func(PanicInfo), phase string, r any) *PanicError {
	stack := debug.Stack()
	firePanicHooks(onPanic, phase, r, stack)
	return &PanicError{Phase: phase, Value: r, Stack: stack}
}

// callHook runs fn and, if it panics, fires OnPanic and re-panics the recovered
// value wrapped in a *PanicError. If the recovered value is already a
// *PanicError (a nested propagation), it is re-panicked as-is without firing
// OnPanic again. The re-panicked *PanicError is converted into a returned error
// by recoverToError (sync entry points) or into stream.Err() by
// recoverToStreamErr (streaming goroutines).
//
// Used only by the propagate-fatal callbacks: OnRequest, OnResponse,
// OnStepFinish, OnFinish, OnBeforeStep, and the StopWhen predicate.
func callHook(onPanic []func(PanicInfo), phase string, fn func()) {
	defer func() {
		if r := recover(); r != nil {
			// Already a *PanicError (re-panicked from an inner callHook): pass it
			// through unchanged so OnPanic is not fired twice.
			if pe := asPanicError(r); pe != nil {
				panic(pe)
			}
			panic(newPanicError(onPanic, phase, r))
		}
	}()
	fn()
}

// asPanicError reports whether a recover() value is (or wraps) a *PanicError,
// returning it if so. It bridges recover()'s any to errors.As: a non-error
// panic value (e.g. a string) yields nil.
func asPanicError(r any) *PanicError {
	if err, ok := r.(error); ok {
		var pe *PanicError
		if errors.As(err, &pe) {
			return pe
		}
	}
	return nil
}

// recoverToError is deferred at synchronous entry points (GenerateText,
// GenerateObject, StreamText, StreamObject). It converts any panic into the
// named return error: a *PanicError passes through; any other value is wrapped
// with newPanicError(phase="internal") and reported to OnPanic. This catch-all
// (consistent with recoverToStreamErr) keeps a panic carrying sensitive data
// from being printed to stderr as an uncaught crash dump.
func recoverToError(onPanic []func(PanicInfo), err *error) {
	if r := recover(); r != nil {
		if pe := asPanicError(r); pe != nil {
			*err = pe
			return
		}
		*err = newPanicError(onPanic, "internal", r)
	}
}

// recoverToStreamErr is deferred at streaming goroutine boundaries (StreamText,
// StreamObject). It converts a *PanicError panic into a stored stream error via
// set so it is reported through stream.Err(). A non-*PanicError value is wrapped
// with newPanicError(phase) so a callback panic in a background goroutine never
// crashes the process.
func recoverToStreamErr(onPanic []func(PanicInfo), phase string, set func(error)) {
	if r := recover(); r != nil {
		if pe := asPanicError(r); pe != nil {
			set(pe)
			return
		}
		set(newPanicError(onPanic, phase, r))
	}
}
