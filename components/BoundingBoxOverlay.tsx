
import React from 'react';
import { BoundingBox } from '../types';

interface BoundingBoxOverlayProps {
  boxes: BoundingBox[];
}

const BoundingBoxOverlay: React.FC<BoundingBoxOverlayProps> = ({ boxes }) => {
  return (
    <div className="absolute inset-0 pointer-events-none z-20">
      {boxes.map((box) => {
        const left = box.xmin / 10;
        const top = box.ymin / 10;
        const width = (box.xmax - box.xmin) / 10;
        const height = (box.ymax - box.ymin) / 10;

        return (
          <div
            key={box.id}
            className="absolute transition-all duration-300 ease-out will-change-[top,left,width,height]"
            style={{
              top: `${top}%`,
              left: `${left}%`,
              width: `${width}%`,
              height: `${height}%`,
            }}
          >
            {/* Box Fill & Border Effect */}
            <div className="absolute inset-0 border border-cyan-400/30 bg-cyan-400/5 rounded-sm shadow-[0_0_20px_rgba(34,211,238,0.2)]"></div>

            {/* Corner Markers */}
            <div className="absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 border-cyan-400 rounded-tl-sm" />
            <div className="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-cyan-400 rounded-tr-sm" />
            <div className="absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-cyan-400 rounded-bl-sm" />
            <div className="absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 border-cyan-400 rounded-br-sm" />

            {/* Label */}
            <div className="absolute -top-6 left-0 flex items-center transform scale-x-[-1] origin-bottom-left">
              <div className="bg-cyan-500/90 text-slate-950 text-[9px] font-black px-2 py-0.5 rounded-sm uppercase tracking-wider shadow-lg backdrop-blur-sm whitespace-nowrap">
                {box.label}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default BoundingBoxOverlay;
