import { Navigate, Route, Routes } from "react-router-dom";
import { EditorPage } from "./pages/EditorPage";
import { ImportPage } from "./pages/ImportPage";
import "./App.css";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<ImportPage />} />
      <Route path="/editor/:name" element={<EditorPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
