import "./theme.css";

import { ChatPanel } from "./components/ChatPanel";
import { Header } from "./components/Header";
import { SubgraphPanel } from "./components/SubgraphPanel";

export default function App() {
  return (
    <div className="app">
      <Header />
      <div className="body">
        <SubgraphPanel />
        <ChatPanel />
      </div>
    </div>
  );
}
