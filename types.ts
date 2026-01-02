
export interface BoundingBox {
  id: string;
  label: string;
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
  timestamp: number;
}

export interface TranscriptionEntry {
  role: 'user' | 'model';
  text: string;
}

export enum ConnectionStatus {
  DISCONNECTED = 'DISCONNECTED',
  CONNECTING = 'CONNECTING',
  CONNECTED = 'CONNECTED',
  ERROR = 'ERROR'
}
