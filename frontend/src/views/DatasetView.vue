<template>
  <div class="dashboard">
    <h2>활용 데이터 개요</h2>
    <div class="header">
      <div
        class="Explanationview"
        :class="{ active: activeView === 'explain' }"
        @click="showExplanation"
      >
        📊데이터 출처 및 가공
      </div>
      <div
        class="tableview"
        :class="{ active: activeView === 'table' }"
        @click="showTable"
      >
        🧾데이터 전처리
      </div>
      <div
        class="apiview"
        :class="{ active: activeView === 'api' }"
        @click="showapi"
      >
        🔗API 샘플
      </div>
    </div>

    <!-- 데이터 출처 및 가공 -->
    <div v-if="activeView === 'explain'" class="explain-container">
      <!-- 데이터 출처 -->
      <div>
        <h2>데이터 출처</h2>
        <p>• 분석 데이터는 <a href="https://www.kaggle.com/datasets/nphantawee/pump-sensor-data/data" target="_blank">Kaggle</a>에서 취득하여 전처리 과정을 거쳐 분석 수행</p>
      </div>
      <br>
      <!--데이터 현황-->
      <div>
        <h2>데이터 요약</h2>
        <div class="toggles">
          <div class="toggle-item" :class="{ expanded: expandedIndex === 0 }">
            <p @click="toggleDetail(0)">
              <a>{{ expandedIndex === 0 ? '▲' : '▼' }}</a>
              &nbsp;시계열 기반 데이터
            </p>
            <div v-if="expandedIndex === 0" class="detail" v-highlight>
              날짜(timestamp): 2018-04-01 00:00:00 ~ 2018-08-31 23:59:00, 1분 단위​
              <div v-if="loadingtimestamp" class="loading">
                <div class="spinner-circle"></div>
                데이터 불러오는 중...
              </div>
              <table v-else-if="timestampData.data.length" class="simple-table">
                <thead>
                  <tr>
                    <th v-for="(h, i) in timestampData.headers" :key="i">{{ h }}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(row, i) in timestampData.data" :key="i">
                    <td v-for="(val, j) in row" :key="j">{{ val }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div class="toggle-item" :class="{ expanded: expandedIndex === 1 }">
            <p @click="toggleDetail(1)">
              <a>{{ expandedIndex === 1 ? '▲' : '▼' }}</a>
              &nbsp;51개의 센서 데이터
            </p>
            <div v-if="expandedIndex === 1" class="detail" v-highlight>
              센서 데이터(sensor_00~sensor_51) : 측정된 Data 값, float
              <div v-if="loadingsensor" class="loading">
                <div class="spinner-circle"></div>
                데이터 불러오는 중...
              </div>
                <table v-if="sensorData.data.length" class="simple-table">
                  <thead>
                    <tr>
                      <th v-for="(h, i) in sensorData.headers" :key="i">{{ h }}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(row, i) in sensorData.data" :key="i">
                      <td v-for="(val, j) in row" :key="j">{{ val }}</td>
                    </tr>
                  </tbody>
                </table>
            </div>
          </div>

          <div class="toggle-item" :class="{ expanded: expandedIndex === 2 }">
            <p @click="toggleDetail(2)">
              <a>{{ expandedIndex === 2 ? '▲' : '▼' }}</a>
              &nbsp;설비 고장 발생 시점 존재
            </p>
            <div v-if="expandedIndex === 2" class="detail" v-highlight>
              설비 상태(machine_status) 3종류 존재
              <div v-if="loadingstatus" class="loading">
                <div class="spinner-circle"></div>
                데이터 불러오는 중...
              </div>
                <table v-if="statusData.data.length" class="simple-table">
                  <thead>
                    <tr>
                      <th v-for="(h, i) in statusData.headers" :key="i">{{ h }}</th>
                      <th>수량</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(count, status) in statusData.counts" :key="status">
                      <td>{{ status }}</td>
                      <td>{{ count }}</td>
                    </tr>
                  </tbody>
                </table>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 데이터 전처리 -->
    <div v-if="activeView === 'table'" class="table-container">
      <!--데이터 전처리-->
      <div>
        <h2>데이터 전처리</h2>
        <div class="toggles">
          <div class="toggle-item" :class="{ expanded: expandedIndex === 3 }">
            <p @click="toggleDetail(3)">
              <a>{{ expandedIndex === 3 ? '▲' : '▼' }}</a>
              &nbsp;불필요한 컬럼 제거         
            </p>
            <div v-if="expandedIndex === 3" class="detail" v-highlight>
              <p>• 모든 값이 0 또는 NaN인 컬럼</p>
              <p>• 데이터 변동성이 낮은 컬럼</p>
              <p style="font-weight: bold;">※결과: 'sensor_00', 'sensor_18', 'sensor_15' 제외</p>
              <pre><code class="language-python">all_zero_cols = data.columns[(data == 0).all()].tolist()
all_nan_cols = data.columns[data.isna().all()].tolist()
threshold = 1.2
low_std_cols = data.std(numeric_only=True).loc[lambda x: x < threshold].index.tolist()
useless_columns = list(set(all_zero_cols + all_nan_cols + low_std_cols))
data = data.drop(columns=useless_columns)</code></pre>
            </div>
          </div>

          <div class="toggle-item" :class="{ expanded: expandedIndex === 4 }">
            <p @click="toggleDetail(4)">
              <a>{{ expandedIndex === 4 ? '▲' : '▼' }}</a>
              &nbsp;설비 상태 숫자 인코딩
            </p>
            <div v-if="expandedIndex === 4" class="detail" v-highlight>
              <p>• machine_status 컬럼을 다음과 같이 수치로 변환</p>
              <pre><code class="language-python">status_map = {'NORMAL': 0, 'RECOVERING': 1, 'BROKEN': 2}
data['machine_status_encoded'] = data['machine_status'].map(status_map)</code></pre>
            </div>
          </div>

          <div class="toggle-item" :class="{ expanded: expandedIndex === 5 }">
            <p @click="toggleDetail(5)">
              <a>{{ expandedIndex === 5 ? '▲' : '▼' }}</a>
              &nbsp;시간 처리 및 보간
            </p>
            <div v-if="expandedIndex === 5" class="detail" v-highlight>
              <p>• timestamp 컬럼을 datetime 타입으로 변환</p>
              <p>• timestamp를 인덱스로 설정</p>
              <p>• 시간 기반 보간(interpolation) 수행</p>
              <pre><code class="language-python">data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)
data = data.interpolate(method='time')</code></pre>
            </div>
          </div>

          <div class="toggle-item" :class="{ expanded: expandedIndex === 6 }" @click="toggleDetail(6)">
            <p>
              <a>{{ expandedIndex === 6 ? '▲' : '▼' }}</a>
              &nbsp;센서 데이터 정규화
            </p>
            <div v-if="expandedIndex === 6" class="detail" v-highlight>
              <p>• StandardScaler를 사용하여 Z-score 정규화</p>
              <pre><code class="language-python">sensor_cols = [col for col in data.columns if col.startswith('sensor')]
scaler = StandardScaler()
data[sensor_cols] = scaler.fit_transform(data[sensor_cols])</code></pre>
            </div>
          </div>
        </div>
      </div>
      <br>
      <!--데이터 분석 준비-->
      <div>
        <h2>데이터 분석 준비</h2>
        <div class="toggles">
          <div class="toggle-item" :class="{ expanded: expandedIndex === 7 }">
            <p @click="toggleDetail(7)">
              <a>{{ expandedIndex === 7 ? '▲' : '▼' }}</a>
              &nbsp;Train 데이터 분리
            </p>
            <div v-if="expandedIndex === 7" class="detail" v-highlight>
              <p>• 전체 데이터 중 60% 학습, 40% 테스트에 활용</p>
              <pre><code class="language-python">total_rows = len(data)
split_idx = int(total_rows * 0.6)
train_data = data.iloc[:split_idx]
test_data = data.iloc[split_idx:]
</code></pre>
            </div>
          </div>

          <div class="toggle-item" :class="{ expanded: expandedIndex === 8 }">
            <p @click="toggleDetail(8)">
              <a>{{ expandedIndex === 8 ? '▲' : '▼' }}</a>
              &nbsp;고장 예측을 위한 윈도우 생성
            </p>
            <div v-if="expandedIndex === 8" class="detail" v-highlight>
              <p>• 시간적 패턴 학습 기반 상태 예측을 위한 슬라이딩 윈도우 생성</p>
              <p>• 센서 데이터를 입력(X), 이후 시점의 설비 상태를 라벨(y)로 구성</p>
              <p>• Train/Test 데이터 60개를 기반으로 다음 시점의 설비 상태를 예측하는 구조</p>
              <pre><code class="language-python">def create_windows_fast(data, window_size=60, step=1, label_col='machine_status_encoded'):
    sensor_cols = [col for col in data.columns if col.startswith('sensor')]
    sensor_data = data[sensor_cols].to_numpy(dtype=np.float32)
    label_data = data[label_col].to_numpy(dtype=np.int8)

    num_windows = (len(sensor_data) - window_size) // step
    X = np.empty((num_windows, window_size, len(sensor_cols)), dtype=np.float32)
    y = np.empty(num_windows, dtype=np.int8)

    for i in range(num_windows):
        idx = i * step
        X[i] = sensor_data[idx:idx + window_size]
        y[i] = label_data[idx + window_size]

    return X, y

X_train, y_train = create_windows_fast(train_data)
X_test, y_test = create_windows_fast(test_data)</code></pre>
            </div>
          </div>

          <div class="toggle-item" :class="{ expanded: expandedIndex === 9 }">
            <p @click="toggleDetail(9)">
              <a>{{ expandedIndex === 9 ? '▲' : '▼' }}</a>
              &nbsp;품질 예측을 위한 윈도우 생성
            </p>
            <div v-if="expandedIndex === 9" class="detail" v-highlight>
              <p>• 센서 데이터의 미래 값을 예측하기 위한 슬라이딩 윈도우 생성</p>
              <p>• 일정 구간의 센서 데이터를 입력(X), 다음 시점의 센서 값을 라벨(y)로 구성</p>
              <p>• Train/Test 데이터 60개를 기반으로 61번째 값을 예측하는 구조</p>
              <pre><code class="language-python">def create_sensor_trend_windows(data, sensor_cols, window_size=60, step=1):
    sensor_data = data[sensor_cols].to_numpy(dtype=np.float32)
    num_windows = (len(sensor_data) - window_size) // step

    X = np.empty((num_windows, window_size, len(sensor_cols)), dtype=np.float32)
    y = np.empty((num_windows, len(sensor_cols)), dtype=np.float32)

    for i in range(num_windows):
        idx = i * step
        X[i] = sensor_data[idx:idx + window_size]
        y[i] = sensor_data[idx + window_size]
    return X, y

X_train, y_train = create_sensor_trend_windows(train_data, sensor_cols)
X_test, y_test = create_sensor_trend_windows(test_data, sensor_cols)</code></pre>
            </div>
          </div>

          <div class="toggle-item" :class="{ expanded: expandedIndex === 10 }">
            <p @click="toggleDetail(10)">
              <a>{{ expandedIndex === 10 ? '▲' : '▼' }}</a>
              &nbsp;고장 유형 분류를 위한 윈도우 생성
            </p>
            <div v-if="expandedIndex === 10" class="detail" v-highlight>
              <p>• 고장 패턴을 분류하기 위한 슬라이딩 윈도우 생성</p>
              <p>• 고장 직전 정상 상태와, 고장 시점의 데이터를 case_{i}로 넘버링</p>
              <p>• 이상 징후 포함 고장 상태 데이터들을 군집화하여 유사 fault type 도출</p>
              <pre><code class="language-python">def extract_pre_failure_windows(broken_times, df, window=10):
    segments = []
    used_broken_times = []

    for i, t in enumerate(broken_times):
        temp_normal = df[df['timestamp'] < t].tail(window)

        temp_broken = df[df['timestamp'] == t]

        if len(temp_normal) == window and len(temp_broken) == 1 and 'BROKEN' not in temp_normal['machine_status'].values:
            combined = pd.concat([temp_normal, temp_broken])
            combined = combined.copy()
            combined['label'] = '1'
            combined['case_id'] = f"case_{i}"
            segments.append(combined)
            used_broken_times.append(t)

    return pd.concat(segments).reset_index(drop=True), used_broken_times
</code></pre>
            </div>
          </div>
        </div>
      </div>
    </div>

      <!-- API 코드 예시 -->
    <div v-if="activeView === 'api'" class="api-container">
      <div>
        <h2>분석 데이터 Web 연동</h2>
        <p>• Python 기반의 FastAPI를 활용</p>
        <p>• 학습 모델, 지표, 결과 데이터를 웹으로 시각화하여 전달하는 기능 수행</p>
        <p>• 오류 처리 등 수업 내 학습한 주요 내용을 활용하여 API 구성</p>
        <div class="toggles">
          <div class="toggle-item" :class="{ expanded: expandedIndex === 11 }">
            <p @click="toggleDetail(11)">
              <a>{{ expandedIndex === 11 ? '▲' : '▼' }}</a>
              &nbsp;예: 센서 품질 예측 신뢰도
            </p>
            <div v-if="expandedIndex === 11" class="detail" v-highlight>
              <p>• def: 재사용 가능한 API 함수 생성 및 입력 파라미터 지정</p>
              <p>• try/except: 파일 및 데이터 오류를 대비한 예외 처리</p>
              <p>• if: 센서명 불일치 시 오류 반환 문구</p>
              <pre><code class="language-python">@app.get("/api/gru-metrics")
def get_gru_metrics(sensor: str = Query(...)):
    try:
        data = load_json("gru_metrics.json")
        df = pd.DataFrame(data.get("details", []))
        matched = df[df["Sensor"] == sensor]
        if matched.empty:
            raise HTTPException(status_code=404, detail=f"{sensor}에 대한 메트릭 없음")
        r2 = float(matched.iloc[0]["R2"])
        return JSONResponse(content={
            "sensor": sensor,
            "confidence_score": round(r2 * 100, 2)
        })
    except Exception:
        logger.exception("GRU 메트릭 로딩 실패")
        raise HTTPException(status_code=500, detail="GRU 메트릭 로딩 실패")
</code></pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

const activeView = ref('explain')
const expandedIndex = ref(null)

const timestampData = ref({ headers: [], data: [] })
const sensorData = ref({ headers: [], data: [] })
const statusData = ref({ headers: [], data: [], counts: {} })

const loadingtimestamp = ref(false)
const loadingsensor = ref(false)
const loadingstatus = ref(false)

function toggleDetail(index) {
  expandedIndex.value = expandedIndex.value === index ? null : index

  if (index === 0 && timestampData.value.data.length === 0) fetchTimestampData()
  if (index === 1 && sensorData.value.data.length === 0) fetchSensorData()
  if (index === 2 && statusData.value.data.length === 0) fetchStatusData()
}

function fetchTimestampData() {
  loadingtimestamp.value = true
  axios.get("http://localhost:8000/api/timestamp_example")
    .then(res => { timestampData.value = res.data })
    .catch(err => console.error("timestamp 호출 실패:", err))
    .finally(() => { loadingtimestamp.value = false })
}

function fetchSensorData() {
  loadingsensor.value = true
  axios.get("http://localhost:8000/api/sensor_example")
    .then(res => { sensorData.value = res.data })
    .catch(err => console.error("sensor 호출 실패:", err))
    .finally(() => { loadingsensor.value = false })
}

function fetchStatusData() {
  loadingstatus.value = true
  axios.get("http://localhost:8000/api/status_example")
    .then(res => { statusData.value = res.data })
    .catch(err => console.error("status 호출 실패:", err))
    .finally(() => { loadingstatus.value = false })
}

const showExplanation = () => { activeView.value = 'explain' }
const showTable = () => { activeView.value = 'table' }
const showapi = () => { activeView.value = 'api' }
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

.explain-container, .table-container, .api-container {
  padding: 10px;
  font-size: 16px;
  line-height: 1.4;
  background-color: #f9f9f9;
  border: 1px solid #ddd;
  margin-top: 10px;
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

.simple-table {
  margin-top: 10px;
  border-collapse: collapse;
  width: 200px;
}

.simple-table th,
.simple-table td {
  border: 1px solid #ccc;
  padding: 6px 10px;
  text-align: center;
}

.loading {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height: 120px;
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