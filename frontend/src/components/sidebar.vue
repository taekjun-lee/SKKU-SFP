<template>
  <div class="sidebar">
    <router-link to="/dashboard" class="title-link">
      <h2 class="title">SFP-PJT</h2>
    </router-link>

    <ul class="nav">
      <li><router-link to="/dashboard">📊 Dashboard</router-link></li>
      <li><router-link to="/contents">📘 프로젝트 개요</router-link></li>
      <li><router-link to="/dataset">📂 활용 데이터</router-link></li>
      <li><router-link to="/algorithm">🧪 모델 비교</router-link></li>
      <!--<li><router-link to="/code">💻 활용 코드</router-link></li>-->
    </ul>
  </div>
</template>

<style scoped>
.sidebar {
  width: 220px;
  height: 100vh;
  background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
  color: white;
  padding: 20px;
  box-sizing: border-box;
  box-shadow: 2px 0 5px rgba(0, 0, 0, 0.2);
}

.title {
  font-size: 24px;
  margin-bottom: 40px;
  text-align: center;
  font-weight: bold;
  color: #ecf0f1;
  border-bottom: 2px solid #42b983;
  padding-bottom: 10px;
}

.nav {
  list-style: none;
  padding: 0;
  margin: 0;
}

.nav li {
  margin-bottom: 20px;
}

.nav a {
  display: block;
  color: #ecf0f1;
  text-decoration: none;
  font-size: 16px;
  font-weight: 500;
  padding: 10px 12px;
  border-radius: 6px;
  transition: background-color 0.3s, color 0.3s;
}

.nav a:hover {
  background-color: #1abc9c;
  color: white;
}

.nav a.router-link-exact-active {
  background-color: #16a085;
  font-weight: bold;
}
</style>