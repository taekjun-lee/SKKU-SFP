import { createRouter, createWebHistory } from 'vue-router'
import DashboardView from '../views/DashboardView.vue'
import AlgorithmView from '../views/AlgorithmView.vue'
import ContentsView from '../views/ContentsView.vue'
import DatasetView from '../views/DatasetView.vue'
import CodeView from '../views/CodeView.vue'

const routes = [
  { path: '/', redirect: '/contents' },
  { path: '/dashboard', component: DashboardView },
  { path: '/algorithm', component: AlgorithmView },
  { path: '/contents', component: ContentsView },
  { path: '/dataset', component: DatasetView },
  { path: '/code', component: CodeView }
]

const router = createRouter({
  history: createWebHistory('/SKKU-SmartFactory-Project/'),
  routes
})

export default router