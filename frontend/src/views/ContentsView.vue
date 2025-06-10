<template>
  <div class="dashboard">
    <!-- 헤더 -->
    <h2>프로젝트 개요</h2>
    <div class="header">
      <div
        class="explainationview"
        :class="{ active: activeView === 'explain' }"
        @click="showExplaination"
      >
        🎯배경 및 목표
      </div>
      <div
        class="tableview"
        :class="{ active: activeView === 'table' }"
        @click="showTable"
      >
        👥추진 내용
      </div>
    </div>

    <!-- 배경 및 목표 -->
    <div v-if="activeView === 'explain'" class="explain-container">
      <h2>프로젝트 배경 및 해결 과제</h2>
      <p>• 기존의 설비 운영 및 관리 체계는 많은 <a class="nkeyword">사람의 개입</a>이 요구됨</p>
      <p>• AI 기반 데이터 분석 및 처리를 통한 <a class="nkeyword">업무 환경 및 품질의 개선</a>이 필요</p>
        <div class="flow-container">
          <div class="flow-box">설비 알람 발생</div>
          <div class="arrow">→</div>
          <div class="nflow-box">과거 이력 및 원인 분석</div>
          <div class="arrow">→</div>
          <div class="nflow-box">조치 방안 수립</div>
          <div class="arrow">→</div>
          <div class="flow-box">작업 수행 및 안정화</div>
        </div>
      <p>• 데이터 분석 과정에서의 휴먼 에러</p>
      <p>• 엔지니어 개입 필요에 의한 업무 비효율성</p>
      <p>• 분석 과정 내 개인별 역량에 따른 의사결정 편차</p>
      <p>• 이상 징후 조기 감지의 한계</p>
      <p>• 대량 데이터 처리 및 판단의 비효율성</p>
      <br><br>
      <h2>과제 수행 목표</h2>
      <p>• 머신러닝을 통한 업무의 <a class="pkeyword">자동화 및 효율화</a> 실현</p>
      <p>• 데이터 기반 의사결정을 통한 <a class="pkeyword">일관되고 오류 없는 의사결정</a> 실현</p>
        <div class="flow-container">
          <div class="flow-box">설비 알람 발생</div>
          <div class="arrow">→</div>
          <div class="automated-section">
            <div class="automation-label">자동화</div>
            <div class="automation-boxes">
              <div class="pflow-box">과거 이력 및 원인 분석</div>
              <div class="arrow">→</div>
              <div class="pflow-box">조치 방안 수립</div>
            </div>
          </div>
          <div class="arrow">→</div>
          <div class="flow-box">작업 수행 및 안정화</div>
        </div>
    </div>

    <!-- 추진 내용 -->
    <div v-else-if="activeView === 'table'" class="table-container">    
      <h2>팀 구성 및 역할</h2>
      <div class="col-containter">
        <div class="member">
          <a class="pkeyword">이택준</a>
          <p>• Web Application 및 시각화 구현</p>
          <p>• Backend API 설계 및 구현(FastAPI)</p>
          <p>• 데이터 분석 및 ML 기반 이상 탐지 로직 구축</p>
          <p>• 데이터 분석 코드 점검 및 개선</p>
        </div>
        <div class="member">
          <a class="pkeyword">정희주</a>
          <p>• 수집 데이터 정제 및 전처리</p>
          <p>• 데이터 분석 및 ML 기반 이상 탐지 로직 구축</p>
          <p>• 분석 결과 기반 설비 이상 알람 및 조치 제안</p>
          <p>• Web Application UI 및 기능 점검</p>
        </div>
      </div>
      <br>
      <h2>과제 추진 과정</h2>
      <p>• 공개 데이터셋을 활용하여 대표적인 ML 알고리즘 비교 분석 후 최적 모델 적용</p>
      <p>• Rule-based 방식으로 고장 발생 시 대응 방안 제안</p>
      <p>• Python 기반의 데이터 가공, 분석, 연계 및 처리 로직 구현</p>
        <div class="flow-container">
          <div class="flow-box2">데이터 취득</div>
          <div class="arrow">→</div>
          <div class="flow-box2">데이터 분석/전처리</div>
          <div class="arrow">→</div>
          <div class="flow-box2">알고리즘 선정</div>
          <div class="arrow">→</div>
          <div class="flow-box2">결과 도출</div>
          <div class="arrow">→</div>
          <div class="flow-box2">데이터 연계</div>
          <div class="arrow">→</div>
          <div class="flow-box2">대시보드 구축</div>
        </div>
      <br>
      <p>※ 실제 배포의 경우 Backend 서버가 필요한 관계로 로컬 환경 기준으로 구현</p>
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
.pkeyword {
  font-weight: bold;
  color: #37419e;
}

.nkeyword {
  font-weight: bold;
  color: #c95656;
}

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
  font-size: 16px;
  line-height: 1.4;
  background-color: #f9f9f9;
  border: 1px solid #ddd;
  margin-top: 10px;
}

.table-container {
  padding: 10px;
  font-size: 16px;
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

.flow-container {
  display: flex;
  align-items: center;
  gap: 10px;
}

.flow-box {
  background-color: #223947;
  align-items: center;
  justify-content: center;
  display: flex;
  gap: 10px;
  color: white;
  width: 150px;
  padding: 15px 25px;
  border-radius: 12px;
  white-space: nowrap;
}

.flow-box2 {
  background-color: #223947;
  align-items: center;
  justify-content: center;
  display: flex;
  gap: 10px;
  color: white;
  width: 100px;
  padding: 15px 25px;
  border-radius: 12px;
  white-space: nowrap;
}

.nflow-box {
  background-color: #cc7c7c;
  align-items: center;
  justify-content: center;
  display: flex;
  gap: 10px;
  color: white;
  width: 150px;
  padding: 15px 25px;
  border-radius: 12px;
  white-space: nowrap;
}

.pflow-box {
  background-color: #4b70a0;
  align-items: center;
  justify-content: center;
  display: flex;
  gap: 10px;
  color: white;
  width: 150px;
  padding: 15px 25px;
  border-radius: 12px;
  white-space: nowrap;
}

.arrow {
  font-size: 24px;
  color: #223947;
}

.automated-section {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.automation-label {
  position: absolute;
  top: -25px;
  font-weight: bold;
  color: #4b70a0;
}

.automation-boxes {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px;
  position: relative;
  z-index: 1;
}

.automation-boxes::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  border: 2px dashed #4b70a0;
  border-radius: 12px;
  z-index: -1;
  box-sizing: border-box;
  animation: dash-pulse 1.5s infinite ease-in-out;
}

@keyframes dash-pulse {
  0% {
    opacity: 1;
    filter: brightness(100%);
  }
  50% {
    opacity: 0.4;
    filter: brightness(160%);
  }
  100% {
    opacity: 1;
    filter: brightness(100%);
  }
}

.col-containter {
  display: grid;
  grid-template-columns: 400px 400px;
  gap: 6px;
}

.member {
  border: 1px dashed #4b70a0;
  border-radius: 12px;
  padding: 10px;
}
</style>