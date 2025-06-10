<script setup>
import { ref, onMounted, defineEmits } from 'vue'

const emit = defineEmits(['sensor-selected'])

const headers = ref([])
const selectedHeader = ref('')

onMounted(async () => {
  try {
    const response = await fetch('http://localhost:8000/api/headers')
    const result = await response.json()
    if (result.headers) {
      headers.value = result.headers
    }
  } catch (error) {
    console.error('헤더 불러오기 실패:', error)
  }
})

function selectSensor(header) {
  selectedHeader.value = header
  emit('sensor-selected', header)
}
</script>

<template>
  <div class="header">Parameter 목록</div>
  <div class="parameterlist">
    <ul v-if="headers.length">
      <li
        v-for="(header, index) in headers"
        :key="index"
        @click="selectSensor(header)"
        :class="{ selected: selectedHeader === header }"
        style="cursor: pointer;"
      >
        {{ header }}
      </li>
    </ul>
    <p v-else>헤더를 불러오는 중입니다...</p>
  </div>
</template>

<style scoped>
.header {
  font-weight: bold;
  border-bottom: 2px solid #aaa;
}

.parameterlist {
  max-height: 400px;
  overflow-y: auto;
  border-radius: 8px;
}

ul {
  list-style-type: none;
  padding-left: 0;
}

li {
  padding: 6px 8px;
  border-bottom: 1px solid #eee;
  font-family: monospace;
  transition: background-color 0.2s;
  color: black;
}

li.selected {
  background-color: #1abc9c;
  font-weight: bold;
  color: white;
}

li:hover {
  background-color: #cce5ff;
  color: #003366;
}
</style>