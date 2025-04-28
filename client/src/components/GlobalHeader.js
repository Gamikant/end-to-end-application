// src/components/GlobalHeader.js
import React from 'react';
import { NavLink, useLocation, useParams } from 'react-router-dom';

const GlobalHeader = () => {
  const { runId } = useParams();
  const location = useLocation();
  const match = location.pathname.match(/\/runs\/([^/]+)/);
  const currentRunId = runId || (match && match[1]);

  return (
    <header className="global-header">
      <nav className="header-nav">
        <NavLink 
          to="/" 
          className={({ isActive }) => `header-link ${isActive ? 'active' : ''}`}
        >
          <span className="header-logo">ML Platform</span>
        </NavLink>
        <div className="header-tabs">
          <NavLink 
            to="/specs" 
            className={({ isActive }) => `header-link ${isActive ? 'active' : ''}`}
          >
            Model Details
          </NavLink>
          <NavLink
            to={currentRunId ? `/runs/${currentRunId}/features` : "/runs/features"}
            className={({ isActive }) => `header-link ${isActive ? 'active' : ''}`}
          >
            Feature Selection
          </NavLink>
          <NavLink
            to={currentRunId ? `/runs/${currentRunId}/results` : "/runs/results"}
            className={({ isActive }) => `header-link ${isActive ? 'active' : ''}`}
          >
            Test Results
          </NavLink>
        </div>
      </nav>
    </header>
  );
};

export default GlobalHeader;
