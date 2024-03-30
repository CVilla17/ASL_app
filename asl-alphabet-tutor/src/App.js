import logo from "./logo.svg";
import "./App.css";
import CustomWebcam from "./components/webcam"; // import it

function App() {
  return (
    <div className="App">
      <CustomWebcam className="left" />
      <div className="right">INSTRUCTIONS</div>
    </div>
  );
}

export default App;
