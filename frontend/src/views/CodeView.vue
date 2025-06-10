<template>
  <div class="dashboard">
    <h2>프로젝트 개요</h2>
    <div class="header">
      <div
        class="explainationview"
        :class="{ active: activeView === 'explain' }"
        @click="showExplaination"
      >
        📁API
      </div>
      <div
        class="tableview"
        :class="{ active: activeView === 'table' }"
        @click="showTable"
      >
        📁Data 분석
      </div>
    </div>

    <!-- 설명 화면 -->
    <div v-if="activeView === 'explain'" class="explain-container">
      <p>Fast API 기반</p>
    </div>

    <!-- 로딩 -->
    <div v-else-if="activeView === 'table'" class="table-container">
      <p>ㅁㄴㅇ</p>
    </div>

  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const activeView = ref('explain')

function showExplaination() {
  activeView.value = 'explain'
}

function showTable() {
    activeView.value = 'table'
}
</script>

<style scoped>
.dashboard {
  padding: 6px;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
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
  font-size: 14px;
  line-height: 1.4;
  background-color: #f9f9f9;
  border: 1px solid #ddd;
  margin-top: 10px;
}

.table-container {
  padding: 10px;
  font-size: 14px;
  line-height: 1.4;
  background-color: #f9f9f9;
  border: 1px solid #ddd;
  margin-top: 10px;
}

.tableview, .explainationview {
  display: flex;
  margin-left: 10px;
  cursor: pointer;
}

.tableview:hover, .explainationview:hover {
  background-color: #cce5ff;
}
</style>