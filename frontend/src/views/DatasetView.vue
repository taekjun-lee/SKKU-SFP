<template>
  <div class="dashboard">
    <h2>활용 데이터 개요</h2>
    <div class="header">
      <div
        class="explainationview"
        :class="{ active: activeView === 'explain' }"
        @click="showExplaination"
      >
        📊데이터 출처 및 가공
      </div>
      <div
        class="tableview"
        :class="{ active: activeView === 'table' }"
        @click="showTable"
      >
        🧾데이터 조회
      </div>
      <div class="fileview" @click="downloadFile">
        💾다운로드
      </div>
    </div>

    <!-- 설명 화면 -->
    <div v-if="activeView === 'explain'" class="explain-container">
      <h2>데이터 출처</h2>
      <p>• 분석 데이터는 <a href="https://www.kaggle.com/datasets/nphantawee/pump-sensor-data/data" target="_blank">Kaggle</a>에서 취득하여 전처리 과정을 거쳐 활용 가능한 형태로 가공</p>
      <p>• 데이터 샘플 및 다운로드는 상단 메뉴 참고</p>
      <br>
      <h2>데이터 전처리</h2>
      <div class="toggles">
        <div class="toggle-item" @click="toggleDetail(0)">
          <p>• Data가 Null이거나 데이터 변동성이 없는 Column 제외 
            <a>{{ expanded[0] ? '▲' : '▼' }}</a>
          </p>
          <div v-if="expanded[0]" class="detail">
            제외 샘플 데이터 보여줄 것
          </div>
        </div>

        <div class="toggle-item" @click="toggleDetail(1)">
          <p>• 특정 Sensor 데이터의 경우 중복/유사한 트렌드를 가지므로 일부 Column 제외 
            <a>{{ expanded[1] ? '▲' : '▼' }}</a>
          </p>
          <div v-if="expanded[1]" class="detail">
            제외 샘플 데이터 보여줄 것
          </div>
        </div>

        <div class="toggle-item" @click="toggleDetail(2)">
          <p>• 훈련 및 검증의 경우 80/20로 데이터를 분할하여 진행 
            <a>{{ expanded[2] ? '▲' : '▼' }}</a>
          </p>
          <div v-if="expanded[2]" class="detail">
            샘플 데이터 보여줄게 있으면 보여주기
          </div>
        </div>
      </div>
      <br>
      <h2>데이터 분석</h2>
      <p>• ㅁㄴㅇ</p>
    </div>

    <!-- 로딩 -->
    <div v-if="activeView === 'table' && isLoading" class="loading">
      <div class="spinner-circle"></div>
      <p>데이터 로딩 중입니다...</p>
    </div>

    <!-- 테이블 화면 -->
    <div v-else-if="activeView === 'table'" class="table-container">
      <table class="styled-table">
        <thead>
          <tr>
            <th v-for="header in headers" :key="header">{{ header }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, index) in paginatedData" :key="index">
            <td v-for="header in headers" :key="header" :title="row[header]">
              {{ row[header] }}
            </td>
          </tr>
        </tbody>
      </table>

      <div class="pagination">
        <button @click="prevPage" :disabled="currentPage === 1">이전</button>
        <span>{{ currentPage }} / {{ totalPages }}</span>
        <button @click="nextPage" :disabled="currentPage === totalPages">다음</button>
      </div>
    </div>

  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const currentPage = ref(1)
const itemsPerPage = 20
const totalItems = ref(0)
const headers = ref([])
const paginatedData = ref([])
const isLoading = ref(false)
const activeView = ref('explain')
const expanded = ref([false, false, false])

function toggleDetail(index) {
  expanded.value[index] = !expanded.value[index]
}

function showExplaination() {
  activeView.value = 'explain'
}

function downloadFile() {
  const link = document.createElement('a')
  link.href = `${import.meta.env.BASE_URL}sensor.csv`
  link.download = 'sensor.csv'
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

async function fetchPage(page) {
  isLoading.value = true
  try {
    const res = await fetch(
      `http://localhost:8000/api/raw-data?page=${page}&size=${itemsPerPage}`
    )
    const result = await res.json()

    if (result.error) throw new Error(result.error)

    paginatedData.value = result.data
    totalItems.value = result.total
    headers.value = result.data.length > 0 ? Object.keys(result.data[0]) : []
  } catch (e) {
    console.error("데이터 요청 실패:", e)
  } finally {
    isLoading.value = false
  }
}

function showTable() {
  if (activeView.value !== 'table') {
    currentPage.value = 1
    fetchPage(currentPage.value)
    activeView.value = 'table'
  }
}

function nextPage() {
  if (currentPage.value < totalPages.value) {
    currentPage.value++
    fetchPage(currentPage.value)
  }
}

function prevPage() {
  if (currentPage.value > 1) {
    currentPage.value--
    fetchPage(currentPage.value)
  }
}

const totalPages = computed(() => Math.ceil(totalItems.value / itemsPerPage))
</script>

<style scoped>
.dashboard {
  padding: 6px;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.table-container {
  overflow-x: auto;
  margin-top: 6px;
}

.styled-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
  text-align: center;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

.styled-table thead {
  background-color: #333;
  color: white;
}

.styled-table th,
.styled-table td {
  padding: 8px 10px;
  border: 1px solid #ddd;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 80px;
  height: 16px;
}

.styled-table tr:nth-child(even) {
  background-color: #f9f9f9;
}

.status-normal {
  background-color: #e0f7e9;
  color: #2e7d32;
  font-weight: bold;
}

.status-alert {
  background-color: #fff3cd;
  color: #d35400;
  font-weight: bold;
}

.status-error {
  background-color: #fdecea;
  color: #c0392b;
  font-weight: bold;
}

.pagination {
  position: fixed;
  bottom: 10px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 10px;
  padding: 6px 12px;
  border: 1px solid #ccc;
  border-radius: 8px;
  z-index: 10;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
}

.pagination button {
  padding: 4px 8px;
  background-color: #333;
  color: white;
  border: none;
  cursor: pointer;
  border-radius: 6px;
}

.pagination button:disabled {
  background-color: #999;
  cursor: not-allowed;
}

.pagination span {
  font-weight: bold;
}

.loading {
  margin-top: 40px;
  font-size: 18px;
  color: #555;
  text-align: center;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.spinner-circle {
  width: 40px;
  height: 40px;
  border: 6px solid #ddd;
  border-top-color: #333;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 10px;
}

.header {
  display: flex;
  border-bottom: 2px solid #aaa;
  font-size: 16px;
}

.header div {
  margin-left: 10px;
  padding: 6px 10px;
  cursor: pointer;
  color: black;
}

.header div:hover {
  background-color: #cce5ff;
}

.header .active {
  background-color: #d6eaff;
  font-weight: 580;
}

.explain-container {
  padding: 10px;
  font-size: 16px;
  line-height: 1.4;
  background-color: #f9f9f9;
  border: 1px solid #ddd;
  margin-top: 10px;
}

.fileview, .tableview, .explainationview {
  display: flex;
  margin-left: 10px;
  cursor: pointer;
}

.fileview:hover, .tableview:hover, .explainationview:hover {
  background-color: #cce5ff;
}

.table-container {
  overflow-x: auto;
  margin-top: 6px;
  transition: max-height 0.4s ease-in-out;
  max-height: 9999px;
}

.toggle-item {
  margin-bottom: 10px;
  cursor: pointer;
}

.toggle-item p {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.toggle-item a {
  font-size: 14px;
  margin-left: 8px;
  color: #0056b3;
}

.detail {
  padding: 6px 12px;
  background-color: #f2f2f2;
  border-left: 4px solid #007acc;
  font-size: 14px;
  margin-top: 4px;
  transition: all 0.3s ease-in-out;
}
</style>