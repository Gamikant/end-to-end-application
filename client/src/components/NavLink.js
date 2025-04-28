import { NavLink } from 'react-router-dom';

// In RunDetails component
<nav className="run-subnav">
  <NavLink 
    to={`/runs/${runId}/features`}
    className={({ isActive }) => isActive ? 'active' : ''}
  >
    Feature Selection
  </NavLink>
  <NavLink 
    to={`/runs/${runId}/results`}
    className={({ isActive }) => isActive ? 'active' : ''}
  >
    Test Results
  </NavLink>
</nav>
