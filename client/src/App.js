// src/App.js
import { Routes, Route } from 'react-router-dom';
import Home from './components/Home';
import ModelSpecs from './components/ModelSpecs';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/specs" element={<ModelSpecs />} />
    </Routes>
  );
}

export default App;
