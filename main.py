#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi 4B + Sony IMX477 — ArUco-сканер (DICT_4X4_50; решётка 3 × 3)

▪ калибровка: движение только по X (±20 мм, шаг 2 мм) —
  останавливается при ≥ 4 найденных метках; результат
  сохраняется в aruco_calib.json;

▪ «смарт-шаг» 0 .80 × FOV  → ≈ 20 % перекрытия;

▪ склейка только MultiBand (простой стабильный вариант).
"""
import os, json, math, time, threading, logging
import cv2, numpy as np
import serial, serial.tools.list_ports
import customtkinter as ctk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk

# ─────────── константы ────────────────────────────────────────────────
ARUCO_DICT   = cv2.aruco.DICT_4X4_50
MARKER_MM    = 5.3          # сторона чёрного квадрата, мм
CAL_Z_MM     = 83
STEP_FACTOR  = 0.80          # смарт-шаг = 0.80 × FOV  (≈ 20 %)
CONFIG_FILE  = "aruco_calib.json"
DEFAULT_RESOLUTION = "3840x2160"
CENTER_X = 54
CENTER_Y = 110

try: cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError: pass
logging.getLogger("cv2").setLevel(logging.ERROR)

# ─────────── утилиты ──────────────────────────────────────────────────
def list_cams(maxd=10):
    be = cv2.CAP_DSHOW if os.name=="nt" else cv2.CAP_V4L2
    out=[]
    for i in range(maxd):
        cap=cv2.VideoCapture(i,be)
        if cap.isOpened(): out.append(str(i))
        cap.release()
    return out or ["0"]

def snap(idx:int,w=1920,h=1080):
    be = cv2.CAP_DSHOW if os.name=="nt" else cv2.CAP_V4L2
    cap=cv2.VideoCapture(idx,be); cap.set(3,w); cap.set(4,h)
    for _ in range(3): cap.grab()
    ok,fr=cap.read(); cap.release(); return fr if ok else None

def f(v,d=0.0):
    try: return float(str(v.get() if isinstance(v,ctk.StringVar) else v).replace(',','.'))
    except: return d

def parse_resolution(value, fallback=(1920,1080)):
    try:
        w,h=value.lower().split("x")
        return int(w),int(h)
    except Exception:
        return fallback

# ─────────── главное окно ─────────────────────────────────────────────
class Scanner(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Tuposcan ArUco scanner"); self.geometry("1280x960")
        ctk.set_appearance_mode("dark")

        # ── переменные интерфейса ──
        self.com   = ctk.StringVar()
        self.cam   = ctk.StringVar(value="0")
        self.resolution = ctk.StringVar(value=DEFAULT_RESOLUTION)
        self.fovX  = ctk.StringVar(value="30")
        self.fovY  = ctk.StringVar(value="17")
        self.stepX = ctk.StringVar(value="30")
        self.stepY = ctk.StringVar(value="17")
        self.z     = ctk.StringVar(value=str(CAL_Z_MM))   # ← БЫЛА ОШИБКА: value=
        self.feed  = ctk.StringVar(value="1500")
        self.scan_profile = ctk.StringVar()
        self.focus_profile = ctk.StringVar()
        self.scan_name = ctk.StringVar()
        self.scan_width = ctk.StringVar()
        self.scan_height = ctk.StringVar()
        self.focus_name = ctk.StringVar()
        self.focus_z = ctk.StringVar()
        self.focus_fovX = ctk.StringVar()
        self.focus_fovY = ctk.StringVar()
        self.focus_step = ctk.StringVar(value="1")

        # ── служебные ──
        self.ser=None; self.frames=[]
        self.grid_cols=self.grid_rows=0; self.canvas_refs=[]; self.stitched=None

        self._load_config()
        self._build_ui()

    # ─────────── сохранение/загрузка калибровки ───────────
    def _load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE,"r",encoding="utf-8") as fp:
                    d=json.load(fp)
                self.fovX.set(f"{d.get('fovX',f(self.fovX)):.2f}")
                self.fovY.set(f"{d.get('fovY',f(self.fovY)):.2f}")
                if d.get("resolution"):
                    self.resolution.set(d["resolution"])
                self.scan_profiles=d.get("scan_profiles",{})
                self.focus_profiles=d.get("focus_profiles",{})
                self.scan_profile.set(d.get("selected_scan_profile",""))
                self.focus_profile.set(d.get("selected_focus_profile",""))
                self._apply_steps()
            except Exception: pass
        if not hasattr(self,"scan_profiles"):
            self.scan_profiles={
                "Весь стол":{"width":200,"height":200},
                "Плата 95x95":{"width":95,"height":95},
            }
        if not hasattr(self,"focus_profiles"):
            self.focus_profiles={
                "Стандарт":{"z":85,"fovX":30,"fovY":17},
            }
        if not self.scan_profile.get():
            self.scan_profile.set(next(iter(self.scan_profiles)))
        if not self.focus_profile.get():
            self.focus_profile.set(next(iter(self.focus_profiles)))
        self._load_profile_values()

    def _save_config(self):
        try:
            with open(CONFIG_FILE,"w",encoding="utf-8") as fp:
                json.dump({
                    "fovX":f(self.fovX),
                    "fovY":f(self.fovY),
                    "resolution":self.resolution.get(),
                    "scan_profiles":self.scan_profiles,
                    "focus_profiles":self.focus_profiles,
                    "selected_scan_profile":self.scan_profile.get(),
                    "selected_focus_profile":self.focus_profile.get(),
                },fp,indent=2,ensure_ascii=False)
        except Exception: pass

    # ─────────── UI ───────────
    def _build_ui(self):
        top=ctk.CTkFrame(self); top.pack(fill="x",pady=5)
        ctk.CTkLabel(top,text="Сериал порт").pack(side="left")
        self.comb_ports=ctk.CTkComboBox(top,values=[p.device for p in serial.tools.list_ports.comports()],
                        variable=self.com,width=160)
        self.comb_ports.pack(side="left",padx=2)
        ctk.CTkButton(top,text="Обновить",command=self._refresh_ports).pack(side="left",padx=2)
        ctk.CTkButton(top,text="Подключить",command=self._connect).pack(side="left",padx=3)
        ctk.CTkButton(top,text="Home",command=lambda:self._g("G28")).pack(side="left")
        ctk.CTkButton(top,text="Unlock",command=lambda:self._g("M17")).pack(side="left")

        settings=ctk.CTkFrame(self); settings.pack(fill="x",pady=4)
        ctk.CTkLabel(settings,text="Камера").pack(side="left",padx=3)
        ctk.CTkComboBox(settings,values=list_cams(),variable=self.cam,width=80).pack(side="left")
        ctk.CTkLabel(settings,text="Разрешение").pack(side="left",padx=6)
        ctk.CTkComboBox(settings,values=("3840x2160","1920x1080","1280x720"),
                        variable=self.resolution,width=120).pack(side="left")
        ctk.CTkLabel(settings,text="FOV X,Y").pack(side="left",padx=6)
        for v in (self.fovX,self.fovY): ctk.CTkEntry(settings,textvariable=v,width=60).pack(side="left")
        ctk.CTkLabel(settings,text="Step X,Y").pack(side="left",padx=6)
        for v in (self.stepX,self.stepY): ctk.CTkEntry(settings,textvariable=v,width=60).pack(side="left")
        ctk.CTkLabel(settings,text="Z").pack(side="left",padx=6); ctk.CTkEntry(settings,textvariable=self.z,width=60).pack(side="left")
        ctk.CTkLabel(settings,text="F").pack(side="left",padx=6); ctk.CTkEntry(settings,textvariable=self.feed,width=60).pack(side="left")
        ctk.CTkButton(settings,text="Калибровка",fg_color="#119911",
                      command=lambda:threading.Thread(target=self._calibrate,daemon=True).start()
                      ).pack(side="left",padx=8)

        scan_profiles=ctk.CTkFrame(self); scan_profiles.pack(fill="x",pady=4)
        ctk.CTkLabel(scan_profiles,text="Профиль сканирования").pack(side="left",padx=4)
        self.scan_profiles_box=ctk.CTkComboBox(scan_profiles,values=list(self.scan_profiles),
                                               variable=self.scan_profile,width=200,
                                               command=lambda _=None:self._select_scan_profile())
        self.scan_profiles_box.pack(side="left")
        ctk.CTkLabel(scan_profiles,text="Имя").pack(side="left",padx=4)
        ctk.CTkEntry(scan_profiles,textvariable=self.scan_name,width=160).pack(side="left")
        ctk.CTkLabel(scan_profiles,text="Ширина").pack(side="left",padx=4)
        ctk.CTkEntry(scan_profiles,textvariable=self.scan_width,width=70).pack(side="left")
        ctk.CTkLabel(scan_profiles,text="Высота").pack(side="left",padx=4)
        ctk.CTkEntry(scan_profiles,textvariable=self.scan_height,width=70).pack(side="left")
        ctk.CTkButton(scan_profiles,text="Сохранить",command=self._save_scan_profile).pack(side="left",padx=4)
        ctk.CTkButton(scan_profiles,text="Удалить",command=self._delete_scan_profile).pack(side="left",padx=4)

        focus_profiles=ctk.CTkFrame(self); focus_profiles.pack(fill="x",pady=4)
        ctk.CTkLabel(focus_profiles,text="Профиль фокуса").pack(side="left",padx=4)
        self.focus_profiles_box=ctk.CTkComboBox(focus_profiles,values=list(self.focus_profiles),
                                                variable=self.focus_profile,width=200,
                                                command=lambda _=None:self._select_focus_profile())
        self.focus_profiles_box.pack(side="left")
        ctk.CTkLabel(focus_profiles,text="Имя").pack(side="left",padx=4)
        ctk.CTkEntry(focus_profiles,textvariable=self.focus_name,width=160).pack(side="left")
        ctk.CTkLabel(focus_profiles,text="Z").pack(side="left",padx=4)
        ctk.CTkEntry(focus_profiles,textvariable=self.focus_z,width=60).pack(side="left")
        ctk.CTkLabel(focus_profiles,text="FOV").pack(side="left",padx=4)
        ctk.CTkEntry(focus_profiles,textvariable=self.focus_fovX,width=60).pack(side="left")
        ctk.CTkEntry(focus_profiles,textvariable=self.focus_fovY,width=60).pack(side="left")
        ctk.CTkLabel(focus_profiles,text="Шаг Z").pack(side="left",padx=4)
        ctk.CTkEntry(focus_profiles,textvariable=self.focus_step,width=60).pack(side="left")
        ctk.CTkButton(focus_profiles,text="Z +",command=lambda:self._nudge_z(1)).pack(side="left",padx=2)
        ctk.CTkButton(focus_profiles,text="Z -",command=lambda:self._nudge_z(-1)).pack(side="left",padx=2)
        ctk.CTkButton(focus_profiles,text="Сохранить",command=self._save_focus_profile).pack(side="left",padx=4)
        ctk.CTkButton(focus_profiles,text="Удалить",command=self._delete_focus_profile).pack(side="left",padx=4)
        ctk.CTkButton(focus_profiles,text="Проверить фокус",command=self._check_focus).pack(side="left",padx=4)

        ctl=ctk.CTkFrame(self); ctl.pack(fill="x",pady=4)
        ctk.CTkButton(ctl,text="Scan",command=lambda:threading.Thread(target=self._scan,daemon=True).start()
                      ).pack(side="left",padx=5)
        ctk.CTkButton(ctl,text="Save",command=self._save).pack(side="left",padx=5)

        self.pvar=ctk.DoubleVar(); ctk.CTkProgressBar(self,variable=self.pvar).pack(fill="x",pady=3)
        self.canvas=ctk.CTkCanvas(self,bg="#202020"); self.canvas.pack(fill="both",expand=True,padx=10,pady=5)
        self.bind("<Configure>",lambda e:self._draw_grid())

    # ─────────── Serial ───────────
    def _connect(self):
        try:
            if self.ser and self.ser.is_open: self.ser.close()
            self.ser=serial.Serial(self.com.get(),250000,timeout=1); time.sleep(2)
            self.ser.reset_input_buffer(); messagebox.showinfo("Serial","Connected")
        except Exception as e: messagebox.showerror("Serial",str(e))

    def _refresh_ports(self):
        ports=[p.device for p in serial.tools.list_ports.comports()]
        self.comb_ports.configure(values=ports)
        if ports and not self.com.get():
            self.com.set(ports[0])

    def _g(self,cmd):
        if not(self.ser and self.ser.is_open): return False
        self.ser.reset_input_buffer(); self.ser.write((cmd+"\n").encode()); self.ser.flush()
        while True:
            l=self.ser.readline().decode(errors="ignore").strip()
            if not l: continue
            if l.startswith("ok"): return True
            if "error" in l.lower(): return False

    # ─────────── smart-step ───────────
    def _apply_steps(self):
        k=STEP_FACTOR
        self.stepX.set(f"{f(self.fovX)*k:.2f}")
        self.stepY.set(f"{f(self.fovY)*k:.2f}")

    # ─────────── калибровка ───────────
    def _calibrate(self):
        try:
            feed=int(f(self.feed,1500)); z=f(self.z,CAL_Z_MM); cam=int(self.cam.get())
            for c in ("G90","G28","M400",f"G1 Z{z:.2f} F{feed}","M400"):
                if not self._g(c): return

            dict4=cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
            params=cv2.aruco.DetectorParameters()
            ids_seen=set(); px=[]
            cv2.namedWindow("Calib",cv2.WINDOW_NORMAL)

            for dx in range(-20,22,2):           # только по X
                self._g(f"G1 X{dx:.2f} Y0 F{feed}"); self._g("M400"); time.sleep(0.25)
                res=parse_resolution(self.resolution.get(),(1920,1080))
                fr=snap(cam,*res); g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
                c,ids,_=cv2.aruco.detectMarkers(g,dict4,parameters=params)
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(fr,c,ids,(0,255,0))
                    for cc,idv in zip(c,ids.flatten()):
                        ids_seen.add(int(idv))
                        pts=cc.reshape(4,2)
                        side=np.mean([np.linalg.norm(pts[i]-pts[(i+1)%4]) for i in range(4)])
                        px.append(side)
                        xc,yc=pts.mean(0); cv2.putText(fr,str(idv),(int(xc)-7,int(yc)-7),
                                                       0,0.5,(0,255,0),1)
                cv2.putText(fr,f"seen {len(ids_seen)}",(10,fr.shape[0]-15),0,0.7,(255,255,0),2)
                cv2.imshow("Calib",fr); cv2.waitKey(1)
                if len(ids_seen) >= 4: break
            cv2.destroyWindow("Calib")

            if len(px)<4: raise RuntimeError("мало меток")

            pxmm = np.median(px)/MARKER_MM
            h_px,w_px=snap(cam,*parse_resolution(self.resolution.get(),(1920,1080))).shape[:2]
            fx,fy = w_px/pxmm, h_px/pxmm
            self.fovX.set(f"{fx:.2f}"); self.fovY.set(f"{fy:.2f}")
            self._apply_steps(); self._save_config()
            messagebox.showinfo("Calibration",f"FOV  {fx:.2f} × {fy:.2f} мм\nmarkers: {len(ids_seen)}")
        except Exception as e: messagebox.showerror("Calibration",str(e))

    # ─────────── профили ───────────
    def _load_profile_values(self):
        scan=self.scan_profiles.get(self.scan_profile.get())
        if scan:
            self.scan_name.set(self.scan_profile.get())
            self.scan_width.set(str(scan.get("width","")))
            self.scan_height.set(str(scan.get("height","")))
        focus=self.focus_profiles.get(self.focus_profile.get())
        if focus:
            self.focus_name.set(self.focus_profile.get())
            self.focus_z.set(str(focus.get("z","")))
            self.focus_fovX.set(str(focus.get("fovX","")))
            self.focus_fovY.set(str(focus.get("fovY","")))
            self.fovX.set(f"{f(focus.get('fovX',self.fovX)):.2f}")
            self.fovY.set(f"{f(focus.get('fovY',self.fovY)):.2f}")
            self.z.set(f"{f(focus.get('z',self.z)):.2f}")
            self._apply_steps()

    def _select_scan_profile(self):
        self._load_profile_values()
        self._save_config()

    def _select_focus_profile(self):
        self._load_profile_values()
        self._save_config()

    def _save_scan_profile(self):
        name=self.scan_name.get().strip()
        if not name:
            messagebox.showwarning("Профиль","Введите имя профиля")
            return
        width=f(self.scan_width)
        height=f(self.scan_height)
        if width<=0 or height<=0:
            messagebox.showwarning("Профиль","Размеры должны быть > 0")
            return
        self.scan_profiles[name]={"width":width,"height":height}
        self.scan_profile.set(name)
        self.scan_profiles_box.configure(values=list(self.scan_profiles))
        self._save_config()

    def _delete_scan_profile(self):
        name=self.scan_profile.get()
        if name in self.scan_profiles:
            del self.scan_profiles[name]
            if self.scan_profiles:
                self.scan_profile.set(next(iter(self.scan_profiles)))
            else:
                self.scan_profile.set("")
            self.scan_profiles_box.configure(values=list(self.scan_profiles))
            self._load_profile_values()
            self._save_config()

    def _save_focus_profile(self):
        name=self.focus_name.get().strip()
        if not name:
            messagebox.showwarning("Профиль","Введите имя профиля")
            return
        z=f(self.focus_z)
        fovx=f(self.focus_fovX)
        fovy=f(self.focus_fovY)
        if fovx<=0 or fovy<=0:
            messagebox.showwarning("Профиль","FOV должен быть > 0")
            return
        self.focus_profiles[name]={"z":z,"fovX":fovx,"fovY":fovy}
        self.focus_profile.set(name)
        self.focus_profiles_box.configure(values=list(self.focus_profiles))
        self._load_profile_values()
        self._save_config()

    def _delete_focus_profile(self):
        name=self.focus_profile.get()
        if name in self.focus_profiles:
            del self.focus_profiles[name]
            if self.focus_profiles:
                self.focus_profile.set(next(iter(self.focus_profiles)))
            else:
                self.focus_profile.set("")
            self.focus_profiles_box.configure(values=list(self.focus_profiles))
            self._load_profile_values()
            self._save_config()

    def _nudge_z(self,sign):
        if not(self.ser and self.ser.is_open):
            messagebox.showwarning("Serial","not connected"); return
        step=f(self.focus_step,1.0)
        z=f(self.focus_z,0.0)+(step*sign)
        self.focus_z.set(f"{z:.2f}")
        self.z.set(f"{z:.2f}")
        self._g(f"G90")
        self._g(f"G1 X{CENTER_X:.2f} Y{CENTER_Y:.2f} Z{z:.2f} F{int(f(self.feed,1500))}")

    def _check_focus(self):
        if not(self.ser and self.ser.is_open):
            messagebox.showwarning("Serial","not connected"); return
        z=f(self.focus_z,CAL_Z_MM)
        self.z.set(f"{z:.2f}")
        for c in ("G90",f"G1 X{CENTER_X:.2f} Y{CENTER_Y:.2f} Z{z:.2f} F{int(f(self.feed,1500))}","M400"):
            if not self._g(c): return

    # ─────────── сетка ───────────
    def _build_grid(self):
        prof=self.scan_profiles.get(self.scan_profile.get(),{"width":0,"height":0})
        x0,y0=0.0,0.0
        x1,y1=prof.get("width",0.0),prof.get("height",0.0)
        sx,sy=f(self.stepX),f(self.stepY)
        cols=int((x1-x0)/sx+1.0001); rows=int((y1-y0)/sy+1.0001)
        xs=x0+np.arange(cols)*sx; ys=y0+np.arange(rows)*sy
        self.positions=[(c,r,xs[c],y) for r,y in enumerate(ys)
                        for c in (range(cols) if r%2==0 else reversed(range(cols)))]
        self.grid_cols,self.grid_rows=cols,rows

    # ─────────── сканирование ───────────
    def _scan(self):
        try:
            if not(self.ser and self.ser.is_open):
                messagebox.showwarning("Serial","not connected"); return
            self._build_grid(); self.frames.clear(); self._draw_grid()
            feed=int(f(self.feed,1500)); z=f(self.z,CAL_Z_MM); cam=int(self.cam.get())
            for c in ("G90","G28","M400",f"G1 Z{z:.2f} F{feed}","M400"):
                if not self._g(c): return
            tot=len(self.positions); self.pvar.set(0)
            res=parse_resolution(self.resolution.get(),(1920,1080))
            for i,(col,row,x,y) in enumerate(self.positions):
                if not self._g(f"G1 X{x:.2f} Y{y:.2f} F{feed}"): continue
                self._g("M400"); time.sleep(0.2); fr=snap(cam,*res)
                if fr is not None: self.frames.append((fr.copy(),col,row)); self._thumb(fr,col,row)
                self.pvar.set((i+1)/tot); self.update_idletasks()
            messagebox.showinfo("Scan","done")
        except Exception as e: messagebox.showerror("Scan",str(e))

    # ─────────── thumb/grid ───────────
    def _draw_grid(self):
        self.canvas.delete("all"); self.canvas_refs.clear()
        c,r=self.grid_cols,self.grid_rows
        if not c or not r: return
        W,H=self.canvas.winfo_width(),self.canvas.winfo_height()
        ar=f(self.fovX)/f(self.fovY); cw=min(W/c, H*ar/r); ch=cw/ar
        tw,th=cw*c,ch*r; ox,oy=(W-tw)/2,(H-th)/2; self._geom=(cw,ch,ox,oy)
        for i in range(c+1): self.canvas.create_line(ox+i*cw,oy,ox+i*cw,oy+th,fill="#444")
        for j in range(r+1): self.canvas.create_line(ox,oy+j*ch,ox+tw,oy+j*ch,fill="#444")
        for fr,c0,r0 in self.frames: self._thumb(fr,c0,r0)

    def _thumb(self,fr,c,r):
        if not hasattr(self,"_geom"): return
        cw,ch,ox,oy=self._geom; inv=self.grid_rows-1-r; x0,y0=ox+c*cw, oy+inv*ch
        h,w=fr.shape[:2]; ar=w/h
        if ar>cw/ch: nw,nh=cw,cw/ar
        else: nw,nh=ch*ar,ch
        t=cv2.resize(fr,(int(nw),int(nh)))
        im=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(t,cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(x0+(cw-nw)/2,y0+(ch-nh)/2,anchor="nw",image=im)
        self.canvas_refs.append(im)

    # ─────────── склейка ───────────
    def _stitch_multiband(self):
        fx,fy=f(self.fovX),f(self.fovY); sx,sy=f(self.stepX),f(self.stepY)
        cols,rows=self.grid_cols,self.grid_rows; h,w=self.frames[0][0].shape[:2]
        ppx,ppy=w/fx,h/fy; W=int((cols-1)*sx*ppx+w); H=int((rows-1)*sy*ppy+h)
        blender=cv2.detail_MultiBandBlender(); blender.setNumBands(5); blender.prepare((0,0,W,H))
        for fr,c,r in self.frames:
            blender.feed(fr.astype(np.int16),255*np.ones(fr.shape[:2],np.uint8),
                         (int(c*sx*ppx),int((rows-1-r)*sy*ppy)))
        pano,_=blender.blend(None,None); self.stitched=cv2.convertScaleAbs(pano)

    # ─────────── save ───────────
    def _save(self):
        self._stitch_multiband()
        if self.stitched is None:
            messagebox.showwarning("Stitch","panorama failed"); return
        p=filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPEG","*.jpg;*.jpeg")])
        if p: cv2.imwrite(p,self.stitched); messagebox.showinfo("Saved",p)

# ─────────── run ───────────
if __name__=="__main__":
    Scanner().mainloop()
