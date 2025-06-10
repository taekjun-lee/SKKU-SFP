<template>
  <div class="layout">
    <Sidebar />
    <main class="main-content">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import Sidebar from './components/sidebar.vue'
</script>

<style scoped>
.layout {
  display: flex;
  height: 100vh;
  width: 100vw;
  margin: 0;
  padding: 0;
}

.main-content {
  flex: 1;
  padding: 6px;
  overflow-y: auto;
}
</style>
