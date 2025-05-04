import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import ErrorBoundary from './components/ErrorBoundary'
import { PowerProvider } from './context/PowerContext'
import { SliderProvider } from './context/SliderContext'
import { ModelProvider } from './context/ModelContext'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <ModelProvider>
        <PowerProvider>
          <SliderProvider>
            <App />
          </SliderProvider>
        </PowerProvider>
      </ModelProvider>
    </ErrorBoundary>
  </React.StrictMode>,
)