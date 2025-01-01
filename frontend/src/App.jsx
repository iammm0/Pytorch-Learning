// src/App.js
// eslint-disable-next-line no-unused-vars
import React, { useState } from 'react';

function App() {
  const [inputText, setInputText] = useState('');
  const [responseText, setResponseText] = useState('');

  // 处理用户输入
  const handleChange = (e) => {
    setInputText(e.target.value);
  };

  // 调用 FastAPI 接口
  const handleSubmit = async () => {
    const response = await fetch('http://127.0.0.1:8000/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: inputText }),
    });
    const data = await response.json();
    setResponseText(data.processed_text);
  };

  return (
    <div>
      <h1>Text Processing App</h1>
      <input type="text" value={inputText} onChange={handleChange} />
      <button onClick={handleSubmit}>Process Text</button>
      {responseText && (
        <div>
          <h2>Processed Text:</h2>
          <p>{responseText}</p>
        </div>
      )}
    </div>
  );
}

export default App;
