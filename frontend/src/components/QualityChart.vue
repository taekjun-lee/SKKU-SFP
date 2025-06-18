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

const actualVisible = ref(false)
const predictedVisible = ref(true)
const verticalLines = ref([])

// ✅ 회복 타임스탬프 수직선 fetch
const fetchRecoveringTimestamps = async () => {
  try {
    const res = await fetch("http://localhost:8000/api/recovering-regions")
    const data = await res.json()
    verticalLines.value = data[0].recovering_timestamps.map(t => new Date(t).getTime() / 1000)
  } catch (e) {
    console.error('Failed to load recovering timestamps', e)
  }
}

// ✅ 수직선 표시 플러그인 (뒤에 깔림)
const verticalLinePlugin = {
  hooks: {
    drawClear: (u) => {
      const ctx = u.ctx
      const { top, height } = u.bbox
      ctx.save()
      ctx.strokeStyle = 'yellow'
      ctx.globalAlpha = 0.3
      ctx.lineWidth = 1
      verticalLines.value.forEach(ts => {
        const x = u.valToPos(ts, 'x', true)
        ctx.beginPath()
        ctx.moveTo(x, top)
        ctx.lineTo(x, top + height)
        ctx.stroke()
      })
      ctx.restore()
    }
  }
}

// ✅ 마우스 휠 줌
const wheelZoomPlugin = {
  hooks: {
    ready: (u) => {
      u.over.addEventListener('wheel', (e) => {
        e.preventDefault()
        const factor = e.deltaY < 0 ? 0.9 : 1.1
        const { min, max } = u.scales.x
        const mouseX = u.posToVal(e.offsetX, 'x')
        const newMin = mouseX - (mouseX - min) * factor
        const newMax = mouseX + (max - mouseX) * factor
        u.setScale('x', { min: newMin, max: newMax })
      }, { passive: false })
    }
  }
}

// ✅ 중간 클릭 팬 기능
const panPlugin = {
  hooks: {
    ready: (u) => {
      let isPanning = false
      let startX = 0, origMin = 0, origMax = 0
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
        const pxPerVal = u.bbox.width / (origMax - origMin)
        const dVal = dx / pxPerVal
        u.setScale('x', { min: origMin - dVal, max: origMax - dVal })
      })
      window.addEventListener('mouseup', () => {
        isPanning = false
      })
    }
  }
}

// ✅ 센서 선택 시 시각화
watch(() => props.selectedSensor, async (newSensor) => {
  if (!newSensor) return
  isLoading.value = true

  await fetchRecoveringTimestamps()

  const response = await fetch(`http://localhost:8000/api/gru-plot?sensor=${encodeURIComponent(newSensor)}`)
  const result = await response.json()

  if (!result.train || !result.test || result.predicted.length === 0) {
    isLoading.value = false
    return
  }

  const trainTs = result.train.map(d => new Date(d[0]).getTime() / 1000)
  const trainVals = result.train.map(d => d[1])
  const testTs = result.test.map(d => new Date(d[0]).getTime() / 1000)
  const testActual = result.test.map(d => d[1])
  const testPred = result.predicted
  const timestamps = [...trainTs, ...testTs]
  const seriesTrain = [...trainVals, ...new Array(testActual.length).fill(null)]
  const seriesActual = [...new Array(trainVals.length).fill(null), ...testActual]
  const seriesPred = [...new Array(trainVals.length).fill(null), ...testPred]

  const data = [timestamps, seriesTrain, seriesActual, seriesPred]

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
      { label: 'Train', stroke: 'blue', width: 1 },
      { label: 'Actual', stroke: 'black', width: 1, show: actualVisible.value },
      { label: 'Predicted', stroke: 'red', width: 1, dash: [5, 3] }
    ],
    axes: [{ show: true }, { show: true }],
    plugins: [wheelZoomPlugin, panPlugin, verticalLinePlugin]
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

const toggleSeriesVisibility = () => {
  if (plotInstance) {
    if (plotInstance.series[2]) plotInstance.series[2].show = actualVisible.value
    if (plotInstance.series[3]) plotInstance.series[3].show = predictedVisible.value
    plotInstance.redraw()
  }
}
</script>

<template>
  <div>
    <div v-if="selectedSensor">
      <div v-if="isLoading" class="loading">
        <div class="spinner-circle"></div>
        <p>데이터를 불러오는 중입니다...</p>
      </div>
      <div v-show="!isLoading" ref="plotDiv" class="chart-container">
        <div class="checkbox">
          <label>
            <input type="checkbox" v-model="actualVisible" @change="toggleSeriesVisibility" />
            Actual
          </label>
          <br>
          <label>
            <input type="checkbox" v-model="predictedVisible" @change="toggleSeriesVisibility" />
            Predicted
          </label>
        </div>
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

.checkbox {
  position: absolute;
  right: 80px;
  top: 10px;
  z-index: 10;
  font-size: 14px;
  line-height: 30px;
  font-weight: bold;
  text-align: left;
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