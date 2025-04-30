import { Routes, Route, Navigate } from 'react-router-dom';
import Home from './components/Home';
import ModelSpecs from './components/ModelSpecs';
import RunDetails from './components/RunDetails';
import GlobalHeader from './components/GlobalHeader';
import PreviousRunsHeader from './components/PreviousRunsHeader';

function App() {
  return (
    <>
      <GlobalHeader />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/specs" element={<ModelSpecs />} />
        {/* These will render RunDetails which handles /features and /results */}
        <Route path="/runs/:runId/*" element={<RunDetails />} />
        <Route path="/previous-runs" element={<PreviousRunsHeader />} />
        {/* If someone tries to go directly to features/results without runId, show error */}
        <Route path="/runs/features" element={<InvalidRun />} />
        <Route path="/runs/results" element={<InvalidRun />} />
        {/* Redirect any unknown route to home */}
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </>
  );
}

function InvalidRun() {
  return (
    <div style={{ padding: "2rem", textAlign: "center", color: "red" }}>
      Invalid: Please run a model first to view Feature Selection or Test Results.
    </div>
  );
}

export default App;
