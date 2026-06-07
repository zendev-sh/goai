package goai

import "runtime/debug"

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
			// r is the recover() value (any). The direct type assertion is
			// intentional, NOT errors.As: we detect whether THIS exact value is
			// the sentinel we already wrapped, to avoid double-wrapping and
			// re-firing OnPanic. errors.As would wrongly match a panic value that
			// merely wraps a *PanicError and would drop the current phase.
			if pe, ok := r.(*PanicError); ok {
				panic(pe)
			}
			panic(newPanicError(onPanic, phase, r))
		}
	}()
	fn()
}

// recoverToError is deferred at synchronous entry points (GenerateText,
// GenerateObject). It converts a *PanicError panic into the named return error
// and re-panics any other value (genuine runtime panics are not masked).
func recoverToError(err *error) {
	if r := recover(); r != nil {
		if pe, ok := r.(*PanicError); ok {
			*err = pe
			return
		}
		panic(r)
	}
}

// recoverToStreamErr is deferred at streaming goroutine boundaries (StreamText,
// StreamObject). It converts a *PanicError panic into a stored stream error via
// set so it is reported through stream.Err(). A non-*PanicError value is wrapped
// with newPanicError(phase) so a callback panic in a background goroutine never
// crashes the process.
func recoverToStreamErr(onPanic []func(PanicInfo), phase string, set func(error)) {
	if r := recover(); r != nil {
		if pe, ok := r.(*PanicError); ok {
			set(pe)
			return
		}
		set(newPanicError(onPanic, phase, r))
	}
}
