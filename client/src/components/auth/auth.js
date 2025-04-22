// src/components/auth/Auth.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const Auth = () => {
  const [credentials, setCredentials] = useState({ 
    email: '', 
    password: '' 
  });
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      // Replace with your actual API endpoint
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });

      if (!response.ok) {
        throw new Error('Login failed');
      }

      const data = await response.json();
      localStorage.setItem('token', data.token);
      navigate('/');
    } catch (error) {
      console.error('Authentication error:', error);
      alert('Login failed. Please check your credentials.');
    }
  };

  return (
    <div className="auth-container">
      <h2>Login</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="email"
          placeholder="Email"
          value={credentials.email}
          onChange={(e) => 
            setCredentials({...credentials, email: e.target.value})
          }
          required
        />
        
        <input
          type="password"
          placeholder="Password"
          value={credentials.password}
          onChange={(e) => 
            setCredentials({...credentials, password: e.target.value})
          }
          required
        />

        <button type="submit">Login</button>
      </form>
    </div>
  );
};

export default Auth;
