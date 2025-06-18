<template>
  <div class="dashboard">
    <h2>ML 모델 성능 비교</h2>
    <div class="header">
      <div
        class="tab1view"
        :class="{ active: activeView === 'tab1' }"
        @click="setTab('tab1')"
      >
        📁고장 시점 예측
      </div>
      <div
        class="tab2view"
        :class="{ active: activeView === 'tab2' }"
        @click="setTab('tab2')"
      >
        📁미래 품질 예측
      </div>
      <div
        class="tab3view"
        :class="{ active: activeView === 'tab3' }"
        @click="setTab('tab3')"
      >
        📁고장 유형 분류
      </div>
    </div>

    <div v-if="activeView === 'tab1'" class="tab1-container">
      <h2>고장 시점 예측</h2>
      <p>• 설비 상태 분류를 위해 <a>해석력과 속도에 강점을 가진 트리 모델</a>과 <a>시계열 패턴 학습에 특화된</a> 딥러닝 모델을 균형 있게 선정</p>
        <p>• 총 4가지(XGBoost, LightGBM, 1D-CNN, LSTM) 모델을 비교</p>
        <div class="toggles">
          <div class="toggle-item" :class="{ expanded: expandedIndex === 0 }">
            <p @click="toggleDetail(0)">
              <a>{{ expandedIndex === 0 ? '▲' : '▼' }}</a>
              &nbsp;F1-Score Heatmap
            </p>
            <div v-if="expandedIndex === 0" class="detail" v-highlight>
              <p>• 모두 BROKEN Class 예측에는 실패</p>
              <p>• NORMAL → BROKEN → RECOVERING 순서로 상태가 전개되므로,<a>고장은 RECOVERING 직전 시점</a>으로 가정 가능</P>
              <p>• NORMAL, RECOVERY에 대해 아주 우수한 F1-Score를 기록한<a>XGBoost</a>&nbsp;채택</p>
              <div v-if="loading" class="loading">
                <div class="spinner-circle"></div>
                차트 생성중...
              </div>
              <div v-else>
                <div v-html="svgContentF1"></div>
              </div>
              </div>
          </div>
          <div class="toggle-item" :class="{ expanded: expandedIndex === 1 }">
            <p @click="toggleDetail(1)">
              <a>{{ expandedIndex === 1 ? '▲' : '▼' }}</a>
              &nbsp;시점별 상태 예측
            </p>
            <div v-if="expandedIndex === 1" class="detail" v-highlight>
              <p>• F1-Score에서 확인하였듯 Broken은 예측에 실패했으나, RECOVERING 시작 시점을 BROKEN으로 분류 가능</p>
              <div v-if="loading" class="loading">
                <div class="spinner-circle"></div>
                차트 생성중...
              </div>
              <div v-else>
                <div v-html="svgContentPrediction"></div>
              </div>
              </div>
          </div>
          <div class="toggle-item" :class="{ expanded: expandedIndex === 2 }">
            <p @click="toggleDetail(2)">
              <a>{{ expandedIndex === 2 ? '▲' : '▼' }}</a>
              &nbsp;모델 학습 코드
            </p>
            <div v-if="expandedIndex === 2" class="detail" v-highlight>
              <p>• [n_estimators], [max_depth]는 기본값으로 사용</p>
              <p>• [learning_rate] 널리 쓰이는 표준값 0.1 사용</p>
              <p>• [objective] 다중 클래스 분류 문제에서 확률이 아닌 최종 class index를 직접 반환하도록 지정</p>
              <p>• [num_class] NORMAL, RECOVERING, BROKEN의 세 가지 클래스</p>
              <p>• [eval_metric] 다중 클래스 확률 예측 성능 평가에 적합한 로그손실 사용</p>
              <div>
                <div>
                  <pre><code class="language-python">xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='multi:softmax',
    num_class=3,
    use_label_encoder=False,
    eval_metric='mlogloss',
    verbosity=0
)

xgb_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
y_pred = xgb_model.predict(X_test.reshape(X_test.shape[0], -1))</code></pre>
                </div>
              </div>
            </div>
          </div>
        </div>

    </div>

    <div v-else-if="activeView === 'tab2'" class="tab2-container">
      <h2>미래 품질 예측</h2>
      <p>• 트렌드 예측을 위해 <a>속도와 병렬 처리에 강점이 있는 모델</a>과 <a>장기 의존성 및 복잡한 시계열 구조 학습에 강점</a>을 가진 모델군 선정</p>
      <p>• 총 6가지(GRU, BiLSTM, LSTM, 1D-CNN, TCN, Transformer) 모델을 비교</p>
      <div class="toggles">
        <div class="toggle-item" :class="{ expanded: expandedIndex === 3 }">
          <p @click="toggleDetail(3)">
            <a>{{ expandedIndex === 3 ? '▲' : '▼' }}</a>
            &nbsp;모델별 평균 MSE
          </p>
          <div v-if="expandedIndex === 3" class="detail" v-highlight>
            <p>• 학습 및 평가에 필요한 시간을 고려하여 약식으로 평가 진행 (학습 데이터의 10% 활용)</p>
            <p>• 비교적 우수한 성능을 보인 GRU, BiLSTM, LSTM 추가 분석 수행</p>
            <div v-if="loading" class="loading">
              <div class="spinner-circle"></div>
              차트 생성중...
            </div>
            <div v-else>
              <div v-html="svgContentMSE"></div>
            </div>
          </div>
        </div>
        <div class="toggle-item" :class="{ expanded: expandedIndex === 4 }">
          <p @click="toggleDetail(4)">
            <a>{{ expandedIndex === 4 ? '▲' : '▼' }}</a>
            &nbsp;3개 모델 상세 분석(GRU, BiLSTM, LSTM)
          </p>
          <div v-if="expandedIndex === 4" class="detail" v-highlight>
            <p>• 전체 데이터 기반으로 학습 시 GRU가 가장 높은 성능을 보이는 것으로 확인</p>
            <p>• 평균 R²가 약 0.8로 시계열 트렌드를 상당히 잘 설명</p>
            <p>• 낮은 MSE와 MAE는 예측 오차가 작아 정확하고 안정적인 예측 성능을 보임</p>
            <div v-if="loading" class="loading">
              <div class="spinner-circle"></div>
              차트 생성중...
            </div>
            <div v-else>
              <div style="margin-top: 20px" v-html="mseTableFiltered"></div>
            </div>
          </div>
        </div>
        <div class="toggle-item" :class="{ expanded: expandedIndex === 5 }">
          <p @click="toggleDetail(5)">
            <a>{{ expandedIndex === 5 ? '▲' : '▼' }}</a>
            &nbsp;모델 학습 코드
          </p>
          <div v-if="expandedIndex === 5" class="detail" v-highlight>
            <p>• [GRU(128)] 널리 쓰이는 일반적인 크기 사용</p>
            <p>• [loss='mse'] 회귀 문제로, 시계열의 연속적인 센서 값 예측 손실의 경우 MSE 사용</p>
            <p>• [validation_split] 일부를 검증용으로 분할 EarlyStopping을 위한 검증 손실 필요</p>
            <p>• [early_stop] 시간 절약 및 일반화 성능 향상을 위해 검증 손실이 향상되지 않으면 학습 조기 종료</p>
            <div>
              <div>
                <pre><code class="language-python">gru_trend_model = Sequential([
    GRU(128, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(len(sensor_cols))
])
gru_trend_model.compile(
    optimizer='adam',
    loss='mse'
)
gru_trend_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)</code></pre>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div v-else-if="activeView === 'tab3'" class="tab3-container">
      <h2>고장 유형 분류</h2>
      <p>• 고장 유형 패턴을 탐지하기 위해 <a>속도와 단순 패턴 구분에 강점이 있는 모델</a>과 <a>복잡한 패턴 구분에 유리한 확률 기반 모델</a>을 선정</p>
      <p>• 총 2가지(KMeans, GMM) 모델을 비교</p>
      <div class="toggles">
        <div class="toggle-item" :class="{ expanded: expandedIndex === 6 }">
          <p @click="toggleDetail(6)">
            <a>{{ expandedIndex === 6 ? '▲' : '▼' }}</a>
            &nbsp;Silhouette/DaviesBouldin Score
          </p>
          <div v-if="expandedIndex === 6" class="detail" v-highlight>
            <p>• 지표상으로는 n=2 의 성능이 월등하나, 두 개의 군집으로는 목표한 분류 및 조치가 어려움</p>
            <p>• 군집을 형성하는 센서 값 평균 확인 결과 군집 4개 시행 시 명확하게 영향을 주는 센서가 포착되지 않음</p>
            <p>• 현재 데이터셋 기준으로 <a>n=3이 합리적</a>이라 판단</p>
            <div v-if="loading" class="loading">
              <div class="spinner-circle"></div>
              차트 생성중...
            </div>
            <div v-else>
              <div style="margin-top: 20px" v-html="classTableFiltered"></div>
            </div>
          </div>
        </div>
        <div class="toggle-item" :class="{ expanded: expandedIndex === 7 }">
          <p @click="toggleDetail(7)">
            <a>{{ expandedIndex === 7 ? '▲' : '▼' }}</a>
            &nbsp;모델 별 클러스터링 시각화(t-SNE)
          </p>
          <div v-if="expandedIndex === 7" class="detail" v-highlight>
            <p>• t-SNE를 이용해 고차원 데이터를 2차원으로 축소한 후, KMeans와 GMM 클러스터링 결과를 시각화</p>
            <div v-if="loading" class="loading">
              <div class="spinner-circle"></div>
              차트 생성중...
            </div>
            <div v-else>
              <div v-html="svgContentclass"></div>
            </div>
          </div>
        </div>
        <div class="toggle-item" :class="{ expanded: expandedIndex === 8 }">
          <p @click="toggleDetail(8)">
            <a>{{ expandedIndex === 8 ? '▲' : '▼' }}</a>
            &nbsp;모델 학습 코드
          </p>
          <div v-if="expandedIndex === 8" class="detail" v-highlight>
            <p>• [n_clusters=2,3,4] 고장 유형이 명확히 정의되어 있지 않아, 다양한 군집 수에서의 데이터 분포 특성을 비교</p>
            <p>• [random_state=42] 실험 반복 시 결과 비교의 일관성을 유지하기 위해 중심점 초기화 시 랜덤성을 고정</p>
            <div>
              <div>
                <pre><code class="language-python">kmeans2 = KMeans(n_clusters=2, random_state=42)
kmeans2.fit(X_train)
df_train['kmeans_cluster2'] = kmeans2.labels_

kmeans3 = KMeans(n_clusters=3, random_state=42)
kmeans3.fit(X_train)
df_train['kmeans_cluster3'] = kmeans3.labels_

kmeans4= KMeans(n_clusters=4, random_state=42)
kmeans4.fit(X_train)
df_train['kmeans_cluster4'] = kmeans4.labels_</code></pre>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const expandedIndex = ref(null)
const activeView = ref('tab1')
const svgContentF1 = ref('')
const svgContentPrediction = ref('')
const svgContentMSE = ref('')
const svgContentclass = ref('')
const loading = ref(false)
const mseTableFiltered = ref('')
const classTableFiltered = ref('')

function toggleDetail(index) {
  if (expandedIndex.value === index) {
    expandedIndex.value = null
    if (index === 0) svgContentF1.value = ''
    if (index === 1) svgContentPrediction.value = ''
    if (index === 2) svgContentMSE.value = ''
  } else {
    expandedIndex.value = index
    if (activeView.value === 'tab1') {
      if (index === 0) fetchF1()
      else if (index === 1) fetchPrediction()
    } else if (activeView.value === 'tab2') {
      if (index === 3) fetchMSE()
      else if (index === 4) fetchMseTables()
    } else if (activeView.value === 'tab3') {
      if (index === 6) fetchclassTables()
      else if (index === 7) fetchclass()
    }
  }
}
function setTab(tab) {
  activeView.value = tab
  expandedIndex.value = null
  svgContentF1.value = ''
  svgContentPrediction.value = ''
  svgContentMSE.value = ''
}

function fetchF1() {
  loading.value = true
  fetch('http://localhost:8000/api/f1-score-svg')
    .then(res => res.text())
    .then(data => {
      svgContentF1.value = data
    })
    .catch(err => {
      svgContentF1.value = `<p style="color:red">[오류] F1-score 차트를 생성할 수 없습니다.</p>`
      console.error(err)
    })
    .finally(() => {
      loading.value = false
    })
}

function fetchPrediction() {
  loading.value = true
  fetch('http://localhost:8000/api/prediction-svg')
    .then(res => res.text())
    .then(data => {
      svgContentPrediction.value = data
    })
    .catch(err => {
      svgContentPrediction.value = `<p style="color:red">[오류] 예측 차트를 생성할 수 없습니다.</p>`
      console.error(err)
    })
    .finally(() => {
      loading.value = false
    })
}

function fetchMSE() {
  loading.value = true
  fetch('http://localhost:8000/api/mse-barplot-svg')
    .then(res => res.text())
    .then(data => {
      svgContentMSE.value = data
    })
    .catch(err => {
      svgContentMSE.value = `<p style="color:red">[오류] MSE 차트를 생성할 수 없습니다.</p>`
      console.error(err)
    })
    .finally(() => {
      loading.value = false
    })
}

function fetchMseTables() {
  loading.value = true

  fetch('http://localhost:8000/api/model-metrics-summary')
    .then(res => res.json())
    .then(data => {
      mseTableFiltered.value = generateHtmlTable(data, 'GRU / LSTM / BiLSTM 상세 성능 요약')
    })
    .catch(err => {
      mseTableFiltered.value = `<p style="color:red">[오류] /api/model-metrics-summary 로부터 데이터를 불러올 수 없습니다.</p>`
      console.error(err)
    })
    .finally(() => {
      loading.value = false
    })
}

function fetchclass() {
  loading.value = true
  fetch('http://localhost:8000/api/tsne-cluster-plot')
    .then(res => res.text())
    .then(data => {
      svgContentclass.value = data
    })
    .catch(err => {
      svgContentclass.value = `<p style="color:red">[오류] cluster 차트를 생성할 수 없습니다.</p>`
      console.error(err)
    })
    .finally(() => {
      loading.value = false
    })
}

function fetchclassTables() {
  loading.value = true

  fetch('http://localhost:8000/api/clustering-metrics-summary')
    .then(res => res.json())
    .then(data => {
      classTableFiltered.value = generateHtmlTable(data, 'KMeans / GMM 상세 성능 요약')
    })
    .catch(err => {
      classTableFiltered.value = `<p style="color:red">[오류] /clustering-metrics-summary 로부터 데이터를 불러올 수 없습니다.</p>`
      console.error(err)
    })
    .finally(() => {
      loading.value = false
    })
}

function generateHtmlTable(data, title = '') {
  if (!data || data.length === 0) return '<p>데이터가 없습니다.</p>'

  const headers = Object.keys(data[0])
  let html = `<h4>${title}</h4><table border="1" cellpadding="4" cellspacing="0"><thead><tr>`
  html += headers.map(h => `<th>${h}</th>`).join('')
  html += '</tr></thead><tbody>'

  for (const row of data) {
    html += '<tr>' + headers.map(h => `<td>${row[h]}</td>`).join('') + '</tr>'
  }
  html += '</tbody></table>'
  return html
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

.tab1-container, .tab2-container, .tab3-container {
  padding: 10px;
  font-size: 16px;
  line-height: 1.4;
  background-color: #f9f9f9;
  border: 1px solid #ddd;
  margin-top: 10px;
}

.tab1view, .tab2view, .tab3view {
  display: flex;
  margin-left: 10px;
  cursor: pointer;
}

.tab1view:hover, .tab2view:hover, .tab3view:hover {
  background-color: #cce5ff;
}

.toggle-item {
  margin-bottom: 10px;
  cursor: pointer;
}

.toggle-item p {
  display: flex;
  align-items: center;
}

.toggle-item a {
  font-size: 14px;
  margin-left: 8px;
  color: #0056b3;
}

.detail {
  padding: 6px 12px;
  background-color: #f6f6f6;
  border-left: 4px solid #007acc;
  font-size: 14px;
  margin-top: 4px;
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