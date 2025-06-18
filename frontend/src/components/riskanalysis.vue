<template>
  <div class="grid-container">

    <div class="predict-box">
      <div class="predict-title">📉 이상 예측 분석 결과</div>
      <p class="description">
        다음 고장 예측 시점까지 남은 일수를 확인할 수 있습니다.
      </p>
      <div class="timestamp-row">
        <span>현재 시점:</span>
        <input type="date" :min="lastTrainTimestamp.slice(0, 10)" :value="selectedToday" @change="updateSelectedDate" />
      </div>
      <div class="analysis" v-if="nextRecovery">
        <template v-if="daysUntil <= 30 && daysUntil >= 0">
          <p>🛠️ 고장 예측 시점: <a class="danger">{{ nextRecovery }}</a></p>
          <p>약 <a class="danger">{{ daysUntil }}일</a> 후 이상 발생이 예상되므로 <a class="danger">주의</a>가 필요합니다.</p>
        </template>
        <template v-else-if="daysUntil > 30">
          <p>✅ 고장 예측 시점: <a class="safe">{{ nextRecovery }}</a></p>
          <p>30일 이내 이상 발생 가능성은 낮아 <a class="safe">정상</a> 상태입니다.</p>
        </template>
        <template v-else>
          <p>📭 오늘 이후 고장 예측이 없습니다.</p>
        </template>
      </div>
      <div v-else>
        <p>📭 예측된 이상 발생 시점이 없습니다.</p>
      </div>
    </div>

    <div class="predict-box">
      <div class="predict-title">🛠️ 고장 유형 분석 결과</div>
      <p class="description">
        다음 발생할 것으로 예상되는 고장 유형을 확인할 수 있습니다.
      </p>
      <div class="analysis" v-if="nextRecovery">
        <template v-if="daysUntil <= 30 && daysUntil >= 0">
          <p><a class="danger">{{ nextRecovery }}</a> 고장 유형은 <a class="danger">{{ predictedType }}</a>입니다.</p>
            <p>※ 학습 데이터의 최대 중심 거리(임계값)을 초과하면 신규 유형으로 판단</p>
        </template>
        <template v-else>
          <p>📭 30일 이내 고장 유형 예측 없음</p>
        </template>
      </div>
      <div v-else>
        <p>📭 예측된 고장 유형 데이터가 없습니다.</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const lastTrainTimestamp = ref('')
const nextRecovery = ref('')
const daysUntil = ref(0)
const selectedToday = ref('')
const predictedType = ref('')
const allRecoveryList = ref([])
const clusterSummaries = ref([])

const findNextRecoveryAfterToday = (todayStr, recoveryList) => {
  const today = new Date(todayStr)
  return recoveryList.find(ts => new Date(ts) > today) || null
}

const calculateDaysUntil = (todayStr, targetStr) => {
  const d1 = new Date(todayStr)
  const d2 = new Date(targetStr)
  const diff = d2 - d1
  return Math.round(diff / (1000 * 60 * 60 * 24))
}

const findNearestClusterType = () => {
  if (!nextRecovery.value || clusterSummaries.value.length === 0) {
    predictedType.value = ''
    return
  }

  const targetDate = new Date(nextRecovery.value)

  const nearest = clusterSummaries.value.reduce((closest, curr) => {
    const currTime = new Date(curr.timestamp)
    const closestTime = new Date(closest.timestamp)
    return Math.abs(currTime - targetDate) < Math.abs(closestTime - targetDate) ? curr : closest
  })

  if (nearest.is_new_cluster) {
    predictedType.value = `신규 고장 유형 (확률: ${nearest.is_new_cluster_highpercent})`
  } else {
    predictedType.value = `${nearest.kmeans_cluster3}번 유형`
  }
}

const updateSelectedDate = (event) => {
  const newDate = event.target.value
  if (newDate) {
    selectedToday.value = newDate
    const futureNext = findNextRecoveryAfterToday(newDate, allRecoveryList.value)

    if (futureNext) {
      nextRecovery.value = futureNext
      daysUntil.value = calculateDaysUntil(newDate, futureNext)
      findNearestClusterType()
    } else {
      nextRecovery.value = ''
      daysUntil.value = -1
      predictedType.value = ''
    }
  }
}

const fetchFutureRecovering = async () => {
  try {
    const res = await fetch('http://localhost:8000/api/future-recovering')
    const result = await res.json()

    lastTrainTimestamp.value = result.last_train_timestamp
    selectedToday.value = lastTrainTimestamp.value.slice(0, 10)
    allRecoveryList.value = result.future_recovering || []

    const firstValid = findNextRecoveryAfterToday(selectedToday.value, allRecoveryList.value)

    if (firstValid) {
      nextRecovery.value = firstValid
      daysUntil.value = calculateDaysUntil(selectedToday.value, firstValid)
    } else {
      nextRecovery.value = ''
      daysUntil.value = -1
    }

    await fetchClusterSummary()
    findNearestClusterType()
  } catch (e) {
    console.error('API 요청 실패:', e)
  }
}

const fetchClusterSummary = async () => {
  try {
    const res = await fetch('http://localhost:8000/api/kmeans-cluster-summary')
    clusterSummaries.value = await res.json()
  } catch (e) {
    console.error('클러스터 요약 API 요청 실패:', e)
  }
}

onMounted(fetchFutureRecovering)
</script>


<style scoped>
.grid-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin: 30px;
}

.predict-box {
  background-color: #f9fbfc;
  border: 1px solid #ccc;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 1px 1px 6px rgba(0, 0, 0, 0.08);
}

.predict-title {
  font-size: 20px;
  font-weight: bold;
  color: #2c3e50;
  margin-bottom: 10px;
}

.description {
  font-size: 14px;
  color: #666;
  margin-bottom: 16px;
}

.timestamp-row {
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 12px;
}

.now {
  font-weight: bold;
  color: #2980b9;
}

input[type="date"] {
  padding: 4px 8px;
  font-size: 14px;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.analysis {
  font-size: 15px;
  line-height: 1.6;
  color: #444;
  text-align: left;
}

a.danger {
  color: red;
  font-weight: bold;
}

a.safe {
  color: green;
  font-weight: bold;
}
</style>
