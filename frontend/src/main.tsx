import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import { ThemeProvider } from './context/ThemeContext'
import { RoleProvider } from './context/RoleContext'
 
 ReactDOM.createRoot(document.getElementById('root')!).render(
   <React.StrictMode>
    <ThemeProvider>
      <RoleProvider>
        <App />
      </RoleProvider>
    </ThemeProvider>
   </React.StrictMode>,
 )
