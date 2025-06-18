<script setup>
import { ref, onMounted, defineEmits, watch } from 'vue'

const emit = defineEmits(['sensor-selected'])
const headers = ref([])
const selectedHeader = ref('')
const loading = ref(false)

onMounted(async () => {
  loading.value = true
  try {
    const cached = localStorage.getItem('cachedHeaders')
    if (cached) {
      headers.value = JSON.parse(cached)
    } else {
      const response = await fetch('http://localhost:8000/api/headers')
      const result = await response.json()
      if (result.headers) {
        headers.value = result.headers
        localStorage.setItem('cachedHeaders', JSON.stringify(result.headers))
      }
    }
  } catch (error) {
    console.error('헤더 불러오기 실패:', error)
  } finally {
    loading.value = false
  }
})


watch(headers, (newHeaders) => {
  if (newHeaders.length > 0 && !selectedHeader.value) {
    selectSensor(newHeaders[0])
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
    <div v-if="loading" class="loading">
      <div class="spinner-circle"></div>
      파라미터 불러오는 중...
    </div>
    <ul v-else-if="headers.length">
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
    <p v-else class="no-headers">불러올 수 있는 파라미터가 없습니다.</p>
  </div>
</template>

<style scoped>
.header {
  font-weight: bold;
  border-bottom: 2px solid #aaa;
  margin-bottom: 8px;
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

.loading {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height: 150px;
  font-size: 16px;
  font-weight: 600;
  color: #666;
  text-align: center;
}

.spinner-circle {
  width: 30px;
  height: 30px;
  border: 4px solid #ccc;
  border-top: 4px solid #007acc;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 10px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.no-headers {
  color: #999;
  font-style: italic;
  padding: 10px;
}
</style>