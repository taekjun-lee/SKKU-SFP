import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
import router from './router'

import 'highlight.js/styles/vs2015.css'
import hljs from 'highlight.js'

const app = createApp(App)

app.directive('highlight', {
  mounted(el) {
    const blocks = el.querySelectorAll('pre code')
    blocks.forEach((block) => {
      hljs.highlightElement(block)
    })
  }
})

app.use(router).mount('#app')
