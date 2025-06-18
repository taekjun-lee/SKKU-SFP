<script setup>
import { ref, watch } from 'vue'
import VueApexCharts from 'vue3-apexcharts'

const props = defineProps({
  selectedSensor: String
})

const confidence = ref(0)
const loading = ref(false)

const chartOptions = ref({
  chart: {
    type: 'radialBar',
    offsetY: -10
  },
  plotOptions: {
    radialBar: {
      startAngle: -90,
      endAngle: 90,
      hollow: {
        size: '60%'
      },
      dataLabels: {
        name: {
          show: true,
          fontSize: '16px'
        },
        value: {
          formatter: (val) => `${val}%`,
          fontSize: '22px',
          show: true
        }
      }
    }
  },
  labels: ['신뢰도'],
  colors: ['#00E396']
})

const fetchMetrics = async (sensor) => {
  try {
    loading.value = true
    const res = await fetch(`http://localhost:8000/api/gru-metrics?sensor=${sensor}`)
    const result = await res.json()
    confidence.value = result.confidence_score
  } catch (error) {
    console.error("신뢰도 메트릭 로딩 실패:", error)
    confidence.value = 0
  } finally {
    loading.value = false
  }
}

watch(() => props.selectedSensor, (newSensor) => {
  if (newSensor) fetchMetrics(newSensor)
}, { immediate: true })
</script>

<template>
  <div class="gauge-container" v-if="selectedSensor">
    <h3>{{ selectedSensor }} 신뢰도</h3>
    <p>R²(결정계수)를 기반으로 0~100%로 환산된 값</p>
    <div v-if="loading" class="loading">
      <div class="spinner-circle"></div>
      신뢰도 로딩 중...
    </div>

    <VueApexCharts
      v-else
      width="300"
      type="radialBar"
      :options="chartOptions"
      :series="[confidence]"
    />
  </div>
</template>

<style scoped>
.gauge-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 10px;
  line-height: 5px;
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
</style>
