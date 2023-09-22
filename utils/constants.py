#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:05:43 2023

@author: patomareao
"""

colors = ["#E77C8D", "#5AA9A2"]
colors_gnome = {
    "blues": ["#99c1f1", "#62a0ea", "#3584e4", "#1c71d8", "#1a5fb4"],
    "greens": ["#8ff0a4", "#57e389", "#33d17a", "#2ec27e", "#26a269"],
    "yellows": ["#f9f06b", "#f8e45c", "#f6d32d", "#f5c211", "#e5a50a"],
    "oranges": ["#ffbe6f", "#f8e45c", "#ff7800", "#e66100", "#c64600"],
    "reds": ["", "", "", "", ""],
    "purples": ["#dc8add", "#c061cb", "#9141ac", "#813d9c", "#613583"],
    "greys": ["#ffffff", "#f6f5f4", "#deddda", "#c0bfbc", "#9a9996"],
    "blacks": ["", "", "", "", ""],
    "browns": ["", "", "", "", ""]
}


vars_info = {
    # General
    "time": {
        "signal_id": "TimeStamp",
        "label": "Time",
        "label_plotly": "Time"
    },
    "Tamb": {
        "signal_id": "TT-DES-030",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{amb}$",
        "label_plotly": "T<sub>amb</sub>"
    },
    "I": {
        "signal_id": "RE-SF-001",
        "units_scada": '',
        "units_model": 'W/m²',
        "label": r"$I$",
        "label_plotly": "I"
    },

    # Thermal storage
    "Tts_h_t": {
        "signal_id": "TT-SF-004",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{ts,h,t}$",
        "label_plotly": "T_<sub>ts,h,t</sub>"
    },
    "Tts_h_m": {
        "signal_id": "TT-SF-005",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{ts,h,m}$",
        "label_plotly": "T_<sub>ts,h,m</sub>"
    },
    "Tts_h_b": {
        "signal_id": "TT-SF-006",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{ts,h,b}$",
        "label_plotly": "T_<sub>ts,h,b</sub>"
    },
    "Tts_c_t": {
        "signal_id": "TT-SF-001",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{ts,c,t}$",
        "label_plotly": "T_<sub>ts,c,t</sub>"
    },
    "Tts_c_m": {
        "signal_id": "TT-SF-002",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{ts,c,m}$",
        "label_plotly": "T_<sub><ts,c,m/sub>"
    },
    "Tts_c_b": {
        "signal_id": "TT-SF-003",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{ts,c,b}$",
        "label_plotly": "T_<sub>ts,c,b</sub>"
    },
    "Tts_t_in": {
        "signal_id": "TT-SF-008",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{ts,t,in}$",
        "label_plotly": "T_<sub>ts,t,in</sub>"
    },
    "Tts_b_in": {
        "signal_id": "TT-AQU-109",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{ts,b,in}$",
        "label_plotly": "T_<sub>ts,b,in</sub>"
    },
    "mts_src": {
        "signal_id": "FT-SF-001",
        "units_scada": '',
        "units_model": 'm³/h',
        "label": r"$m_{ts,src}$",
        "label_plotly": "q_<sub>ts,src</sub>"
    },
    "mts_dis": {
        "signal_id": "FT-AQU-101",
        "units_scada": '',
        "units_model": 'm³/h',
        "label": r"$m_{ts,dis}$",
        "label_plotly": "q_<sub>ts,dis</sub>"
    },
    
    # Three-way valve
    "T_3wv_src2": {
        "signal_id": "TT-SF-004",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{3wv,src \: (TS)}$",
        "color": "#b85450",
        "label_plotly": "T_<sub>3wv,src</sub>"
    },
    "T_3wv_dis_in": {
        "signal_id": "TT-AQU-107a",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{3wv,dis,in}$",
        "color": "#dd8452"
    },
    "T_3wv_dis_out": {
        "signal_id": "HW1TT21",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{3wv,dis,out \: (MED)}$",
        "color": "#7a98c4"
    },
    "m_3wv_dis": {
        "signal_id": "FT-AQU-100",
        "units_scada": '',
        "units_model": 'm³/h',
        "label": r"$m_{3wv,dis}$"
    },
    "m_3wv_src": {
        "signal_id": "FT-AQU-101",
        "units_scada": '',
        "units_model": 'm³/h',
        "label": r"$m_{3wv,src}$",
        "color": "#b85450"
    },
    "R_3wv": {
        "signal_id": "ZC-AQU-TCV102",
        "units_scada": '',
        "units_model": r'%',
        "label": r"$R_{3wv}$"
    },
    "T_3wv_dis_out2": {
        "signal_id": "TT-AQU-109",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{3wv,dis,out}$",
        "color": "#7a98c4"
    },
    "T_3wv_src": {
        "signal_id": "TT-AQU-106",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{3wv,src}$",
        "color": "#b85450"
    },
    
    
    # Solar field
    "q_sf_l1": {
        "signal_id": "FT-SF-003",
        "units_scada": '',
        "units_model": 'm³/h',
        "label_plotly": "q<sub>sf,l1</sub>",
        "label": r"$q_{sf,l1}$"
    },
    "Tsf_l2_in": {
        "signal_id": "TT-SF-018",
        "units_scada": '',
        "units_model": 'ºC',
        "label_plotly": "T<sub>sf,l2,in</sub>",
        "label": r"$T_{sf,l2,in}$"
    },
    "Tsf_l2_out": {
        "signal_id": "TT-SF-019",
        "units_scada": '',
        "units_model": 'ºC',
        "label_plotly": "T<sub>sf,l2,out</sub>",
        "label": r"$T_{sf,l2,out}$"
    },
    "q_sf_l2": {
        "signal_id": "FT-SF-005",
        "units_scada": '',
        "units_model": 'm³/h',
        "label_plotly": "q<sub>sf,l2</sub>",
        "label": r"$q_{sf,l2}$"
    },
    "Tsf_l3_in": {
        "signal_id": "TT-SF-024",
        "units_scada": '',
        "units_model": 'ºC',
        "label_plotly": "T<sub>sf,l3,in</sub>",
        "label": r"$T_{sf,l3,in}$"
    },
    "Tsf_l3_out": {
        "signal_id": "TT-SF-025",
        "units_scada": '',
        "units_model": 'ºC',
        "label_plotly": "T<sub>sf,l3,out</sub>",
        "label": r"$T_{sf,l3,out}$"
    },
    "q_sf_l3": {
        "signal_id": "FT-SF-006",
        "units_scada": '',
        "units_model": 'm³/h',
        "label_plotly": "q<sub>sf,l3</sub>",
        "label": r"$q_{sf,l3}$"
    },
    "Tsf_l4_in": {
        "signal_id": "TT-SF-030",
        "units_scada": '',
        "units_model": 'ºC',
        "label_plotly": "T<sub>sf,l4,in</sub>",
        "label": r"$T_{sf,l4,in}$"
    },
    "Tsf_l4_out": {
        "signal_id": "TT-SF-031",
        "units_scada": '',
        "units_model": 'ºC',
        "label_plotly": "T<sub>sf,l4,out</sub>",
        "label": r"$T_{sf,l4,out}$"
    },
    "q_sf_l4": {
        "signal_id": "FT-SF-007",
        "units_scada": '',
        "units_model": 'm³/h',
        "label_plotly": "q<sub>sf,l4</sub>",
        "label": r"$q_{sf,l4}$"
    },
    "Tsf_l5_in": {
        "signal_id": "TT-SF-036",
        "units_scada": '',
        "units_model": 'ºC',
        "label_plotly": "T<sub>sf,l5,in</sub>",
        "label": r"$T_{sf,l5,in}$"
    },
    "Tsf_l5_out": {
        "signal_id": "TT-SF-037",
        "units_scada": '',
        "units_model": 'ºC',
        "label_plotly": "T<sub>sf,l5,out</sub>",
        "label": r"$T_{sf,l5,out}$"
    },
    "q_sf_l5": {
        "signal_id": "FT-SF-008",
        "units_scada": '',
        "units_model": 'm³/h',
        "label_plotly": "q<sub>sf,l5</sub>",
        "label": r"$q_{sf,l5}$"
    },
    "q_sf": {
        "signal_id": "FT-SF-002",
        "units_scada": '',
        "units_model": 'm³/h',
        "label_plotly": "q<sub>sf</sub>",
        "label": r"$q_{sf}$"
    },
    "Tsf_in": {
        "signal_id": "TT-SF-009",
        "units_scada": '',
        "units_model": 'ºC',
        "label_plotly": "T<sub>sf,in</sub>",
        "label": r"$T_{sf,in}$"
    },
    "Tsf_out": {
        "signal_id": "TT-SF-010",
        "units_scada": '',
        "units_model": 'ºC',
        "label_plotly": "T<sub>sf,out</sub>",
        "label": r"$T_{sf,out}$"
    },
    
    # MED
    "Tmed_s_in": {
        "signal_id": "TT-AQU-algo",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{med,s,in}$",
        "label_plotly": "T<sub>med,s,in</sub>",
    },
    "Tmed_c_out": {
        "signal_id": "completar",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{med,c,out}$",
        "label_plotly": "T<sub>med,c,out</sub>",
    },
    "mmed_d": {
        "signal_id": "completar?",
        "units_scada": '',
        "units_model": 'm³/h',
        "label": r"$q_{med,d}$",
        "label_plotly": "q<sub>med,d</sub>",
    },
    "mmed_s": {
        "signal_id": "completar?",
        "units_scada": '',
        "units_model": 'm³/h',
        "label": r"$q_{med,s}$",
        "label_plotly": "q<sub>med,s</sub>",
    },
    "mmed_f": {
        "signal_id": "completar?",
        "units_scada": '',
        "units_model": 'm³/h',
        "label": r"$q_{med,f}$",
        "label_plotly": "q<sub>med,f</sub>",
    },
    "mmed_c": {
        "signal_id": "completar?",
        "units_scada": '',
        "units_model": 'm³/h',
        "label": r"$q_{med,c}$",
        "label_plotly": "q<sub>med,c</sub>",
    },
    "Tmed_s_out": {
        "signal_id": "TT-AQU-algo",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{med,s,out}$",
        "label_plotly": "T<sub>med,s,out</sub>",
    },
    "Tmed_c_in": {
        "signal_id": "?",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{med,c,in}$",
        "label_plotly": "T<sub>med,c,in</sub>",
    },
    "wmed_f": {
        "signal_id": "?",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$w_{med,f}$",
        "label_plotly": "w<sub>med,f</sub>",
    },
    "STEC_med": {
        "signal_id": "?",
        "units_scada": '',
        "units_model": 'kWhth/m³',
        "label": r"$STEC_{med}$",
        "label_plotly": "STEC<sub>med</sub>",
    },
    "SEEC_med": {
        "signal_id": "?",
        "units_scada": '',
        "units_model": 'kWhe/m³',
        "label": r"$SEEC_{med}$",
        "label_plotly": "SEEC<sub>med</sub>",
    },
    
    # Heat exchanger
    "Thx_p_in": {
        "signal_id": "TT-SF-010",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{hx,p,in}$",
        "label_plotly": "T<sub>hx,p,in</sub>"
    },
    "Thx_s_out": { # Invertido temporalmente!!
        "signal_id": "TT-SF-009",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{hx,s,out}$",
        "label_plotly": "T<sub>hx,s,out</sub>"
    },
    "Thx_s_in": {
        "signal_id": "TT-SF-007",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{hx,s,in}$",
        "label_plotly": "T<sub>hx,s,in</sub>"
    },
    "Thx_p_out": { # Invertido temporalmente!!
        "signal_id": "TT-SF-008",
        "units_scada": '',
        "units_model": 'ºC',
        "label": r"$T_{hx,p,out}$",
        "label_plotly": "T<sub>hx,p,out</sub>"
    },
    "mhx_p": {
        "signal_id": "FT-SF-002",
        "units_scada": 'L/min',
        "units_model": 'm³/h',
        "label": r"$q_{hx,p}$",
        "label_plotly": "q<sub>hx,p</sub>"
    },
    "mhx_s": {
        "signal_id": "FT-SF-001",
        "units_scada": 'L/min',
        "units_model": 'm³/h',
        "label": r"$q_{hx,s}$",
        "label_plotly": "q<sub>hx,s</sub>"
    },
}




# Habría que invertir keys por values para que no haya nombres duplicados
var_names = {
    # General
    "time": "TimeStamp",
    "Tamb": "TT-DES-030",
    
    # Thermal storage
    "Tts_h_t": "TT-SF-004",
    "Tts_h_m": "TT-SF-005",
    "Tts_h_b": "TT-SF-006",
    "Tts_c_t": "TT-SF-001",
    "Tts_c_m": "TT-SF-002",
    "Tts_c_b": "TT-SF-003",
    "Tts_t_in": "TT-SF-008",
    "Tts_b_in": "TT-AQU-109",
    "m_ts_src": "FT-SF-001",
    "m_ts_dis": "FT-AQU-101",
             
    # Three way valve
    "T_3wv_src":"TT-SF-004", 
    "T_3wv_dis_in":"TT-AQU-107a",
    "T_3wv_dis_out":"HW1TT21", 
    "m_3wv_dis":"FT-AQU-100",
    "m_3wv_src": "FT-AQU-101", 
    "R_3wv": "ZT-AQU-TCV102",
    
    # Heat exchanger
    
    
    # Solar field
}

var_labels = {
    # Thermal storage
    "Tts_h_t": r"$T_{ts,h,t}$",
    "Tts_h_m": r"$T_{ts,h,m}$",
    "Tts_h_b": r"$T_{ts,h,b}$",
    "Tts_c_t": r"$T_{ts,c,t}$",
    "Tts_c_m": r"$T_{ts,c,m}$",
    "Tts_c_b": r"$T_{ts,c,b}$",
    "m_ts_dis": r"$\dot{m}_{dis}$",
    "m_ts_src": r"$\dot{m}_{src}$",
    "Tts_t_in": r"$T_{ts,t,in}$",
    
    # Three way valve
    "T_3wv_src": r"$T_{3wv,src}$",
    "T_3wv_dis_in": r"$T_{3wv,dis,in}$",
    "T_3wv_dis_out": r"$T_{3wv,dis,out}$",
    "m_3wv_dis": r"$\dot{m}_{3wv,dis}$",
    "m_3wv_src": r"$\dot{m}_{3wv,src}$",
    "R_3wv": r"$R_{3wv}$",
    
    # Heat exchanger
    
    
    # Solar field
}