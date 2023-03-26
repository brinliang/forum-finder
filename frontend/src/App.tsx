import { useState, useEffect } from 'react'
import axios from 'axios'

const App = () => {

  const [textState, setTextState] = useState('Write your question here!')
  const [ratesState, setRatesState] = useState<any[]>([])

  const updateText = (value: string) => {
    setTextState(value)
  }

  const updateRates = () => {

    const requestOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: textState })
    };

    fetch('/api/predict', requestOptions)
    .then((response) => {
      console.log(response)
      return response.json()
    })
    .then((data) => setRatesState(data.rates))
    .catch((error) => console.log(error));
  }


  return (
    <>
      <textarea style={{resize: 'none', width: '400px', height: '100px', fontSize: '20px', padding: '20px', margin: '20px', fontFamily: 'sans-serif'}}
        value={textState} onChange={(e) => updateText(e.target.value)}/>
      <br />
      <button style={{width: '100px', height: '50px', fontSize: '20px', margin: '0px 0px 20px 20px', fontFamily: 'sans-serif'}}
        onClick={updateRates}>submit</button>
      {ratesState.map(r => <div style={{fontSize: '20px', margin: '20px 0px 20px 20px', fontFamily: 'sans-serif'}}
        key={String(r.forum)}> {r.forum}: {r.probability} </div>)}
    </>
  )
}

export default App;