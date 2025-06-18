<template>
  <div class="dashboard">
    <!-- 헤더 -->
    <h2>프로젝트 개요</h2>
    <div class="header">
      <div
        class="Explanationview"
        :class="{ active: activeView === 'explain' }"
        @click="showExplanation"
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
      <div
        class="architectureview"
        :class="{ active: activeView === 'architecture' }"
        @click="showarchitecture"
      >
        🖥️시스템 구성
      </div>
    </div>

    <!-- 배경 및 목표 -->
    <div v-if="activeView === 'explain'" class="explain-container">
      <h2>프로젝트 배경 및 해결 과제</h2>
      <p>• 기존의 설비 운영 및 관리 체계는 많은 <a class="nkeyword">사람의 개입</a>이 요구됨</p>
      <p>• 설비 보전은 사후(BM), 주기(TBM), 조건(CBM)에 의존한 보전 활동 뿐으로 <a class="nkeyword">효율적인 고장 예측</a>이 필요</p>
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
          <p>• Web Application 구현</p>
          <p>• Backend API 설계 및 구현</p>
          <p>• 설비 이상 탐지 시점 예측</p>
          <p>• 미래 품질 트렌드 예측</p>
        </div>
        <div class="member">
          <a class="pkeyword">정희주</a>
          <p>• 데이터 취득/분석/전처리</p>
          <p>• 설비 이상 유형 자동 분류</p>
          <p>• 자료 정리 및 발표</p>
          <p>• 코드 및 UI 기능 점검</p>
        </div>
      </div>
      <br>
      <h2>과제 추진 과정</h2>
      <p>• 공개 데이터셋을 활용하여 대표적인 ML 알고리즘 비교 분석 후 최적 모델 적용</p>
      <p>• ML 기반 <a class="pkeyword">①설비 미래 품질 트렌드, ②고장 발생 시점, ③고장 유형 분석</a></p>
      <p>• Python 기반 데이터 가공, 분석 및 웹 연계 로직 구현</p>
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
    </div>

    <!-- 시스템 구성 -->
    <div v-else="activeView === 'architecture'" class="architecture-container">    
      <h2>주요 파일 및 역할</h2>
      <p>• 3가지 학습 모델에 대한 Python 파일</p>
      <p>• 웹 연동을 위한 API용 Python파일</p>
      <p>• 실제 운영 시에는 Batch 파일을 통해 주기적으로 학습을 수행하고, 업데이트된 정보를 웹으로 제공</p>
      <a>※ 본 프로젝트에서는 별도 Backend 서버가 없으므로 로컬 환경에서 구동</a>
        <div>
          <br>
          <img class="image" src="/architecture.PNG" alt="시스템 구성도" />
        </div>
    </div>

  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
const activeView = ref('explain')

function showExplanation() {
  activeView.value = 'explain'
}

function showTable() {
    activeView.value = 'table'
}

function showarchitecture() {
    activeView.value = 'architecture'
}
</script>

<style scoped>
.image {
  border: 1px dashed rgb(150, 148, 148);
  width: 50%;
}

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

.explain-container, .architecture-container, .table-container {
  padding: 10px;
  font-size: 16px;
  line-height: 1.4;
  background-color: #f9f9f9;
  border: 1px solid #ddd;
  margin-top: 10px;
}

.tableview, .Explanationview, .architectureview {
  display: flex;
  margin-left: 10px;
  cursor: pointer;
}

.tableview:hover, .Explanationview:hover, .architectureview:hover {
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