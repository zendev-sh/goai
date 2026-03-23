<script setup lang="ts">
import { ref, onMounted } from "vue";

interface Version {
  tag: string;
  date: string;
  label: string;
}

const versions = ref<Version[]>([]);
const current = ref("latest");
const latestTag = ref("");
const branchBaseURL = ref("");

onMounted(async () => {
  try {
    const base = import.meta.env.BASE_URL || "/";
    const res = await fetch(`${base}versions.json`);
    if (res.ok) {
      const data = await res.json();
      versions.value = data.versions || [];
      latestTag.value = data.latest || "";
      branchBaseURL.value = data.branchBaseURL || "";

      // Detect current version from URL path (e.g. /v0.4.0/getting-started/)
      const path = window.location.pathname;
      const match = path.match(/^\/(v[\d.]+)\//);
      if (match) {
        current.value = match[1];
      }
    }
  } catch {
    // versions.json not available
  }
});

function navigate(tag: string) {
  const path = window.location.pathname;
  // Strip current version prefix from path
  const stripped = path.replace(/^\/(v[\d.]+)\//, "/");
  if (tag === "latest") {
    if (branchBaseURL.value) {
      // On a branch deploy, go back to main site
      window.location.href = "https://goai.sh" + stripped;
    } else {
      window.location.href = stripped;
    }
  } else {
    // Navigate to versioned branch deploy
    // CF Pages branch names use dashes: v0.4.2 → v0-4-2
    const branch = tag.replace(/\./g, "-");
    if (branchBaseURL.value) {
      const url = branchBaseURL.value.replace("{branch}", branch);
      window.location.href = url + stripped;
    } else {
      window.location.href = "/" + tag + stripped;
    }
  }
}
</script>

<template>
  <div v-if="versions.length > 1" class="version-switcher">
    <select
      :value="current"
      @change="navigate(($event.target as HTMLSelectElement).value)"
    >
      <option value="latest">
        latest{{ latestTag ? ` (${latestTag})` : "" }}
      </option>
      <option v-for="v in versions" :key="v.tag" :value="v.tag">
        {{ v.tag }}
      </option>
    </select>
  </div>
</template>

<style scoped>
.version-switcher {
  margin-left: 8px;
}

.version-switcher select {
  background: transparent;
  color: var(--vp-c-text-2);
  border: 1px solid var(--vp-c-divider);
  padding: 3px 8px;
  font-family: var(--vp-font-family-mono);
  font-size: 0.72rem;
  font-weight: 500;
  cursor: pointer;
  appearance: none;
  -webkit-appearance: none;
  padding-right: 20px;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' fill='none' stroke='%23999' stroke-width='1.5'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 6px center;
}

.version-switcher select:hover {
  color: var(--vp-c-text-1);
  border-color: var(--vp-c-text-3);
}
</style>
