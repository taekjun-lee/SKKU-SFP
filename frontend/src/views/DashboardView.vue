<template>
  <div class="dashboard-grid">
    <div class="header">Parameter Chart</div>
    <div class="parameter">
      <ParameterInfo @sensor-selected="selectedSensor = $event"/>
    </div>
    <div class="graph">
      <QualityChart :selectedSensor="selectedSensor"/>
    </div>
    <div class="header">Data Analysis</div>
    <div class="risk">
      <riskanalysis/>
    </div>
    <div class="algorithm">
      <algorithm/>
    </div>
    <div class="maint">
      <maintanalysis/>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import QualityChart from '../components/QualityChart.vue'
import ParameterInfo from '../components/ParameterInfo.vue'
import riskanalysis from '../components/riskanalysis.vue'
import maintanalysis from '../components/maintanalysis.vue'
import algorithm from '../components/algorithm.vue'
const selectedSensor = ref('')
</script>
<script setup>
</script>

<style scoped>
.dashboard-grid {
  display: grid;
  grid-template-columns: 300px 450px 450px 450px;
  grid-template-rows: auto 450px auto 180px 180px;
  gap: 6px;
}

.dashboard-grid > div {
  border: 1px solid #ccc;
  box-sizing: border-box;
}

.header {
  grid-column: 1 / 5;
  background-color: #2c3e50;
  color: white;
  padding: 6px;
  font-weight: bold;
  text-align: left;
  border-radius: 6px;
  font-size: 20px;
}

.parameter {
  grid-column: 1 / 2;
  grid-row: 2 / 3;
  background-color: white;
  padding: 6px;
  border-radius: 8px;
  text-align: center;
}

.graph {
  grid-column: 2 / 5;
  grid-row: 2 / 3;
  background-color: white;
  padding: 6px;
  border-radius: 8px;
  text-align: center;
}

.risk {
  grid-column: 1 / 4;
  grid-row: 4 / 5;
  background-color: white;
  padding: 6px;
  border-radius: 8px;
  text-align: center;
}

.maint {
  grid-column: 1 / 4;
  grid-row: 5 / 6;
  background-color: white;
  padding: 6px;
  border-radius: 8px;
  text-align: center;
}

.algorithm {
  grid-column: 4 / 4;
  grid-row: 4 / 6;
  background-color: white;
  padding: 10px;
  border-radius: 8px;
  text-align: center;
}
</style>