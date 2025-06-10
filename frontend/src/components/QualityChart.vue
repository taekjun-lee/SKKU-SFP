<script setup>
import { ref, watch, nextTick } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'

const props = defineProps({
  selectedSensor: String
})

const plotDiv = ref(null)
let plotInstance = null
const isLoading = ref(false)

const wheelZoomPlugin = {
  hooks: {
    ready: (u) => {
      u.over.addEventListener('wheel', (e) => {
        e.preventDefault()

        const factor = e.deltaY < 0 ? 0.9 : 1.1
        const { min, max } = u.scales.x
        const range = max - min
        const mouseX = u.posToVal(e.offsetX, 'x')

        const newMin = mouseX - (mouseX - min) * factor
        const newMax = mouseX + (max - mouseX) * factor

        u.setScale('x', { min: newMin, max: newMax })
      }, { passive: false })
    }
  }
}

const panPlugin = {
  hooks: {
    ready: (u) => {
      let isPanning = false
      let startX = 0
      let origMin = 0
      let origMax = 0

      u.over.addEventListener('mousedown', (e) => {
        if (e.button !== 1) return
        isPanning = true
        startX = e.clientX
        origMin = u.scales.x.min
        origMax = u.scales.x.max
      })

      window.addEventListener('mousemove', (e) => {
        if (!isPanning) return
        const dx = e.clientX - startX
        const range = origMax - origMin
        const pxPerVal = u.bbox.width / range
        const dVal = dx / pxPerVal

        u.setScale('x', {
          min: origMin - dVal,
          max: origMax - dVal
        })
      })

      window.addEventListener('mouseup', () => {
        isPanning = false
      })
    }
  }
}

watch(() => props.selectedSensor, async (newSensor) => {
  if (!newSensor) return

  isLoading.value = true

  const now = new Date()
  const end = now.toISOString()
  const start = new Date(now.setDate(now.getDate() - 14)).toISOString()

  const response = await fetch(
    `http://localhost:8000/api/sensor-data?sensor=${encodeURIComponent(newSensor)}&start=${start}&end=${end}`
  )
  const result = await response.json()
  if (!result.timestamps || result.timestamps.length === 0) {
    isLoading.value = false
    return
  }

  const data = [result.timestamps, result.values]

  const opts = {
    width: 1300,
    height: 400,
    cursor: {
      focus: { prox: 30 },
      drag: { x: true, y: false }
    },
    legend: {
      show: true,
      live: true
    },
    scales: {
      x: { time: true },
      y: { auto: true }
    },
    series: [
      {},
      {
        label: 'Value',
        stroke: 'blue',
        width: 1
      }
    ],
    axes: [
      { show: true },
      { show: true }
    ],
    plugins: [wheelZoomPlugin, panPlugin]
  }

  await nextTick()

  const container = plotDiv.value
  if (container.offsetWidth === 0 || container.offsetHeight === 0) {
    setTimeout(() => {
      if (plotInstance) {
        plotInstance.setData(data)
      } else {
        plotInstance = new uPlot(opts, data, container)
      }
      isLoading.value = false
    }, 100)
  } else {
    if (plotInstance) {
      plotInstance.destroy()
    }
    plotInstance = new uPlot(opts, data, container)
    isLoading.value = false
  }
}, { immediate: true })
</script>

<template>
  <div>
    <div v-if="selectedSensor">
      <div v-if="isLoading" class="loading">
        <div class="spinner-circle"></div>
        <p>데이터를 불러오는 중입니다...</p>
      </div>
      <div v-show="!isLoading" ref="plotDiv" class="chart-container">
        <div class="help-icon">
          ?
          <div class="tooltip">
            🔍Zoom: 마우스 스크롤
            <br />
            ↔️Pan: 휠 클릭 드래그
            <br />
            🔄Reset: 더블 클릭
          </div>
        </div>
      </div>
    </div>
    <div v-else class="placeholder">
      <p>📌 Parameter를 선택해주세요</p>
    </div>
  </div>
</template>


<style scoped>
.placeholder, .loading {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height: 400px;
  font-size: 16px;
  font-weight: 600;
  color: #666;
  text-align: center;
  border-radius: 8px;
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
  margin-bottom: 10px;
}

.chart-container {
  position: relative;
}

.help-icon {
  position: absolute;
  top: 10px;
  right: 10px;
  z-index: 10;
  background-color: #f0f0f0;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  font-size: 16px;
  font-weight: bold;
  color: #333;
  text-align: center;
  line-height: 24px;
  cursor: pointer;
  user-select: none;
}

.help-icon:hover .tooltip {
  display: block;
}

.tooltip {
  display: none;
  position: absolute;
  top: 30px;
  right: 0;
  background: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 8px 10px;
  border-radius: 6px;
  font-size: 12px;
  white-space: nowrap;
  z-index: 11;
  line-height: 30px;
  text-align: left;
}
</style>