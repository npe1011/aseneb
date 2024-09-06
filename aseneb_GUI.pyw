import os
import sys
import shutil
import time
import subprocess as sp
from threading import Thread
from pathlib import Path
from typing import Optional, Any

import wx
from wx import xrc
import wx.grid
import wx.lib.newevent
import numpy as np

APP_DIR = (os.path.dirname(os.path.abspath(__file__)))
sys.path.append(APP_DIR)
import config
from aseneb import NEBProjectNonBlocking, NEBResult, SingleTrajectory
from aseneb.utils import remove

AllCalcEndEvent, EVT_ALL_CALC_END = wx.lib.newevent.NewEvent()


class CalcAllThread(Thread):

    def __init__(self,
                 init_structure_file: Path,
                 final_structure_file: Path,
                 neb_project: NEBProjectNonBlocking,
                 parent_window: Any):
        Thread.__init__(self)
        self.init_structure_file = init_structure_file
        self.final_structure_file = final_structure_file
        self.neb_project = neb_project
        self.parent_window: XTBNEBApp = parent_window
        self.terminate_flag = False
        self.start()

    def run(self):

        if not self.terminate_flag:
            self.neb_project.load_init_structure(self.init_structure_file)
            while self.neb_project.check() == 1:
                if self.terminate_flag:
                    self.neb_project.terminate()
                    break
                self.parent_window.update_all()
                time.sleep(config.CHECK_INTERVAL/1000)
        if not self.terminate_flag:
            self.parent_window.logging('Finished: Loading initial structure.')

        if not self.terminate_flag:
            self.neb_project.load_final_structure(self.final_structure_file)
            while self.neb_project.check() == 1:
                if self.terminate_flag:
                    self.neb_project.terminate()
                    break
                self.parent_window.update_all()
                time.sleep(config.CHECK_INTERVAL/1000)
        if not self.terminate_flag:
            self.parent_window.logging('Finished: Loading final structure.')

        # This is a blocking process
        if not self.terminate_flag:
            self.neb_project.interpolate()
            self.parent_window.logging('Finished: Interpolation')

        if not self.terminate_flag:
            self.neb_project.run_neb()
            while self.neb_project.check() == 1:
                if self.terminate_flag:
                    self.neb_project.terminate()
                    break
                self.parent_window.update_all()
                time.sleep(config.CHECK_INTERVAL/1000)
        if not self.terminate_flag:
            self.parent_window.logging('Finished: NEB optimization')

        wx.PostEvent(self.parent_window, AllCalcEndEvent(terminated=self.terminate_flag))

    def terminate(self):
        self.terminate_flag = True


class ProjectFileDropTarget(wx.FileDropTarget):
    def __init__(self, window):
        wx.FileDropTarget.__init__(self)
        self.window = window

    def OnDropFiles(self, x, y, file_names):
        file = Path(file_names[0]).absolute()
        if file.suffix.lower() == 'json':
            self.window.load_project_file(file)
        return True


class TextControlFileDropTarget(wx.FileDropTarget):
    """
    DropTarget to set file full path to text control.
    """
    def __init__(self, window):
        wx.FileDropTarget.__init__(self)
        self.window: wx.TextCtrl = window

    def OnDropFiles(self, x, y, file_names):
        file = Path(file_names[0]).absolute()
        if file.suffix.lower() != 'json':
            self.window.SetValue(str(file))
        return True


class TextViewFrame(wx.Frame):
    def __init__(self, parent, title, text):
        wx.Frame.__init__(self, parent, -1, title)
        self.SetSize((600,400))
        self.init_frame()
        self.text_ctrl_main.SetValue(text)

    def init_frame(self):
        # set controls
        panel = wx.Panel(self, wx.ID_ANY)
        layout = wx.BoxSizer(wx.VERTICAL)
        self.text_ctrl_main = wx.TextCtrl(panel, wx.ID_ANY, style=wx.TE_READONLY | wx.TE_MULTILINE)
        font = wx.Font(12, wx.FONTFAMILY_MODERN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.text_ctrl_main.SetFont(font)
        layout.Add(self.text_ctrl_main, 1, wx.EXPAND | wx.ALL, border=3)
        panel.SetSizerAndFit(layout)


class XTBNEBApp(wx.App):

    def OnInit(self):
        # Current NEBProject
        self.project: Optional[NEBProjectNonBlocking] = None
        self.calc_all_thread: Optional[CalcAllThread] = None

        self.timer = wx.Timer(self)  # Set timer
        self.timer.Start(config.CHECK_INTERVAL)

        self.load_controls()
        self.init_controls()
        self.set_menus()
        self.set_events()

        # Drug & Drop settings
        dt = ProjectFileDropTarget(self)
        self.frame.SetDropTarget(dt)

        self.update_all()

        # redirect
        sys.stdout = self.text_ctrl_log
        sys.stderr = self.text_ctrl_log

        self.frame.Show()
        return True

    def load_controls(self):
        self.resource = xrc.XmlResource('./wxgui/gui.xrc')
        self.frame = self.resource.LoadFrame(None, 'frame')

        self.button_run_all: wx.Button = xrc.XRCCTRL(self.frame, 'button_run_all')

        self.button_project_new: wx.Button = xrc.XRCCTRL(self.frame, 'button_project_new')
        self.button_project_open: wx.Button = xrc.XRCCTRL(self.frame, 'button_project_open')
        self.text_ctrl_project: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_project')
        self.text_ctrl_init_structure_file: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_init_structure_file')
        self.button_init_structure_open: wx.Button = xrc.XRCCTRL(self.frame, 'button_init_structure_open')
        self.checkbox_init_opt: wx.CheckBox = xrc.XRCCTRL(self.frame, 'checkbox_init_opt')
        self.button_init_load: wx.Button = xrc.XRCCTRL(self.frame, 'button_init_load')
        self.text_ctrl_init_result: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_init_result')
        self.button_init_result_view: wx.Button = xrc.XRCCTRL(self.frame, 'button_init_result_view')
        self.text_ctrl_final_structure_file: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_final_structure_file')
        self.button_final_structure_open: wx.Button = xrc.XRCCTRL(self.frame, 'button_final_structure_open')
        self.checkbox_final_opt: wx.CheckBox = xrc.XRCCTRL(self.frame, 'checkbox_final_opt')
        self.button_final_load: wx.Button = xrc.XRCCTRL(self.frame, 'button_final_load')
        self.text_ctrl_final_result: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_final_result')
        self.button_final_result_view: wx.Button = xrc.XRCCTRL(self.frame, 'button_final_result_view')
        self.text_ctrl_num_images: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_num_images')
        self.choice_interpolation_method: wx.Choice = xrc.XRCCTRL(self.frame, 'choice_interpolation_method')
        self.button_interpolation_run: wx.Button = xrc.XRCCTRL(self.frame, 'button_interpolation_run')
        self.text_ctrl_interpolation_result: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_interpolation_result')
        self.button_interpolation_result_view: wx.Button = xrc.XRCCTRL(self.frame, 'button_interpolation_result_view')
        self.choice_neb_method: wx.Choice = xrc.XRCCTRL(self.frame, 'choice_neb_method')
        self.text_ctrl_neb_k: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_neb_k')
        self.checkbox_neb_climb: wx.CheckBox = xrc.XRCCTRL(self.frame, 'checkbox_neb_climb')
        self.choice_neb_optimizer: wx.Choice = xrc.XRCCTRL(self.frame, 'choice_neb_optimizer')
        self.text_ctrl_neb_fmax: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_neb_fmax')
        self.text_ctrl_neb_steps: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_neb_steps')
        self.text_ctrl_neb_parallel: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_neb_parallel')
        self.button_neb_run: wx.Button = xrc.XRCCTRL(self.frame, 'button_neb_run')
        self.list_box_neb_result_files: wx.ListBox = xrc.XRCCTRL(self.frame, 'list_box_neb_result_files')
        self.button_neb_view: wx.Button = xrc.XRCCTRL(self.frame, 'button_neb_view')
        self.button_neb_view_all: wx.Button = xrc.XRCCTRL(self.frame, 'button_neb_view_all')
        self.button_neb_plot: wx.Button = xrc.XRCCTRL(self.frame, 'button_neb_plot')
        self.button_neb_plot_all: wx.Button = xrc.XRCCTRL(self.frame, 'button_neb_plot_all')
        self.button_neb_info: wx.Button = xrc.XRCCTRL(self.frame, 'button_neb_info')
        self.button_neb_info_all: wx.Button = xrc.XRCCTRL(self.frame, 'button_neb_info_all')
        self.button_neb_save_ts: wx.Button = xrc.XRCCTRL(self.frame, 'button_neb_save_ts')
        self.button_neb_delete: wx.Button = xrc.XRCCTRL(self.frame, 'button_neb_delete')
        self.notebook_calculator: wx.Notebook = xrc.XRCCTRL(self.frame, 'notebook_calculator')
        self.choice_xtb_method: wx.Choice = xrc.XRCCTRL(self.frame, 'choice_xtb_method')
        self.text_ctrl_xtb_charge: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_xtb_charge')
        self.text_ctrl_xtb_uhf: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_xtb_uhf')
        self.radio_box_xtb_solvation: wx.RadioBox = xrc.XRCCTRL(self.frame, 'radio_box_xtb_solvation')
        self.choice_xtb_solvent: wx.Choice = xrc.XRCCTRL(self.frame, 'choice_xtb_solvent')
        self.text_ctrl_xtb_cpu: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_xtb_cpu')
        self.text_ctrl_xtb_memory: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_xtb_memory')
        self.button_g16_template_load: wx.Button = xrc.XRCCTRL(self.frame, 'button_g16_template_load')
        self.text_ctrl_g16_template: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_g16_template')
        self.button_g16_template_edit: wx.Button = xrc.XRCCTRL(self.frame, 'button_g16_template_edit')
        self.text_ctrl_g16_cpu: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_g16_cpu')
        self.text_ctrl_g16_memory: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_g16_memory')
        self.text_ctrl_g16_init_guess_keywords: wx.TextCtrl = xrc.XRCCTRL(self.frame,
                                                                          'text_ctrl_g16_init_guess_keywords')
        self.button_g16_init_guess_run: wx.Button = xrc.XRCCTRL(self.frame, 'button_g16_init_guess_run')
        self.text_ctrl_current_calculation: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_current_calculation')
        self.text_ctrl_calculation_log: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_calculation_log')
        self.text_ctrl_log: wx.TextCtrl = xrc.XRCCTRL(self.frame, 'text_ctrl_log')
        self.button_log_delete: wx.Button = xrc.XRCCTRL(self.frame, 'button_log_delete')

        self.button_stop: wx.Button = xrc.XRCCTRL(self.frame, 'button_stop')

        return True

    def init_controls(self):
        self.text_ctrl_init_structure_file.SetDropTarget(TextControlFileDropTarget(self.text_ctrl_init_structure_file))
        self.text_ctrl_final_structure_file.SetDropTarget(TextControlFileDropTarget(self.text_ctrl_final_structure_file))
        for method in config.INTERPOLATION_METHOD_LIST:
            self.choice_interpolation_method.Append(method)
        for method in config.NEB_METHOD_LIST:
            self.choice_neb_method.Append(method)
        for method in config.NEB_OPTIMIZER_LIST:
            self.choice_neb_optimizer.Append(method)
        for method in config.XTB_GFN_LIST:
            self.choice_xtb_method.Append(method)
        for solvent in config.XTB_SOLVENT_LIST:
            self.choice_xtb_solvent.Append(solvent)
        self.load_default_settings()
        self.button_stop.Enable(False)

    def set_menus(self):
        # set menu and event
        menu_settings: wx.Menu = wx.Menu()
        self.menu_settings_notify_finished: wx.MenuItem = \
            menu_settings.AppendCheckItem(wx.ID_ANY, 'Notify when calculation finished')
        if config.DEFAULT_NOTIFY_FINISHED:
            self.menu_settings_notify_finished.Check(True)

        menu_emergency: wx.Menu = wx.Menu()
        self.menu_emergency_reset: wx.MenuItem = \
            menu_emergency.Append(wx.ID_ANY, 'Reset', 'Emergency reset.')

        # set to menu bar
        menu_bar = wx.MenuBar()
        menu_bar.Append(menu_settings, '&Settings')
        menu_bar.Append(menu_emergency, '&Emergency')
        self.frame.SetMenuBar(menu_bar)

    def set_events(self):
        # Buttons
        self.button_run_all.Bind(wx.EVT_BUTTON, self.on_button_run_all)

        self.button_project_new.Bind(wx.EVT_BUTTON, self.on_button_project_new)
        self.button_project_open.Bind(wx.EVT_BUTTON, self.on_button_project_open)

        self.button_init_structure_open.Bind(wx.EVT_BUTTON, self.on_button_init_structure_open)
        self.button_final_structure_open.Bind(wx.EVT_BUTTON, self.on_final_structure_open)
        self.button_init_load.Bind(wx.EVT_BUTTON, self.on_button_init_load)
        self.button_final_load.Bind(wx.EVT_BUTTON, self.on_button_final_load)

        self.button_interpolation_run.Bind(wx.EVT_BUTTON, self.on_button_interpolation_run)

        self.button_neb_run.Bind(wx.EVT_BUTTON, self.on_button_neb_run)

        self.button_g16_template_load.Bind(wx.EVT_BUTTON, self.on_button_g16_template_load)
        self.button_g16_template_edit.Bind(wx.EVT_BUTTON, self.on_button_g16_template_edit)
        self.button_g16_init_guess_run.Bind(wx.EVT_BUTTON, self.on_button_g16_init_guess_run)

        self.button_init_result_view.Bind(wx.EVT_BUTTON, self.on_button_init_result_view)
        self.button_final_result_view.Bind(wx.EVT_BUTTON, self.on_button_final_result_view)

        self.button_interpolation_result_view.Bind(wx.EVT_BUTTON, self.on_button_interpolation_result_view)

        self.button_neb_view.Bind(wx.EVT_BUTTON, self.on_button_neb_view)
        self.button_neb_view_all.Bind(wx.EVT_BUTTON, self.on_button_neb_view_all)
        self.button_neb_plot.Bind(wx.EVT_BUTTON, self.on_button_neb_plot)
        self.button_neb_plot_all.Bind(wx.EVT_BUTTON, self.on_button_neb_plot_all)
        self.button_neb_info.Bind(wx.EVT_BUTTON, self.on_button_neb_info)
        self.button_neb_info_all.Bind(wx.EVT_BUTTON, self.on_button_neb_info_all)
        self.button_neb_save_ts.Bind(wx.EVT_BUTTON, self.on_button_neb_save_ts)
        self.button_neb_delete.Bind(wx.EVT_BUTTON, self.on_button_neb_delete)

        self.button_log_delete.Bind(wx.EVT_BUTTON, self.on_button_log_delete)

        self.button_stop.Bind(wx.EVT_BUTTON, self.on_button_stop)

        # When exit
        self.frame.Bind(wx.EVT_CLOSE, self.on_close)

        # Timer event
        self.Bind(wx.EVT_TIMER, self.on_timer)

        # Emergency reset event (menu)
        self.Bind(wx.EVT_MENU, self.on_emergency_reset, self.menu_emergency_reset)

        # Calc All En Event
        self.Bind(EVT_ALL_CALC_END, self.on_all_calc_end_event)

    def with_timer_stop(func):
        def inner(self, *args, **kwargs):
            if self.calc_all_thread is not None:
                self.timer.Stop()
            result = func(self, *args, **kwargs)
            self.update_all()
            if self.calc_all_thread is not None:
                self.timer.Start(config.CHECK_INTERVAL)
            return result
        return inner

    def logging(self, message: Any) -> None:
        """
        output logs
        """
        log_string = (''.join(message)).rstrip()
        self.text_ctrl_log.write(log_string + '\n')

    def finish_calculation(self) -> None:
        self.enable_calculation_buttons(True)
        if self.menu_settings_notify_finished.IsChecked():
            wx.MessageBox('Calculation Finished', 'Calculation Finished', wx.OK | wx.ICON_INFORMATION)

    def enable_calculation_buttons(self, value: bool) -> None:
        self.button_run_all.Enable(value)
        self.button_init_load.Enable(value)
        self.button_final_load.Enable(value)
        self.button_interpolation_run.Enable(value)
        self.button_neb_run.Enable(value)
        self.button_g16_init_guess_run.Enable(value)
        self.button_stop.Enable(not value)

    def load_default_settings(self) -> None:
        # [2] Input Structures
        self.checkbox_init_opt.SetValue(config.DEFAULT_OPT_INIT)
        self.checkbox_final_opt.SetValue(config.DEFAULT_OPT_FINAL)
        # [3] Interpolation
        self.text_ctrl_num_images.SetValue(str(config.DEFAULT_NUM_IMAGES))
        self.choice_interpolation_method.SetStringSelection(config.DEFAULT_INTERPOLATION_METHOD)
        # [4] NEB Optimization
        self.choice_neb_method.SetStringSelection(config.DEFAULT_NEB_METHOD)
        self.text_ctrl_neb_k.SetValue(str(config.DEFAULT_NEB_K))
        self.checkbox_neb_climb.SetValue(config.DEFAULT_NEB_CLIMB)
        self.choice_neb_optimizer.SetStringSelection(config.DEFAULT_NEB_OPTIMIZER)
        self.text_ctrl_neb_fmax.SetValue(str(config.DEFAULT_NEB_FMAX))
        self.text_ctrl_neb_steps.SetValue(str(config.DEFAULT_NEB_STEPS))
        self.text_ctrl_neb_parallel.SetValue(str(config.DEFAULT_NEB_PARALLEL))
        # Calculator_type
        if config.DEFAULT_CALCULATOR_TYPE.lower() == 'xtb':
            self.notebook_calculator.SetSelection(0)
        elif config.DEFAULT_CALCULATOR_TYPE.lower() == 'g16':
            self.notebook_calculator.SetSelection(1)
        # XTB Settings
        self.choice_xtb_method.SetStringSelection(config.DEFAULT_XTB_GFN)
        self.text_ctrl_xtb_charge.SetValue(str(config.DEFAULT_XTB_CHARGE))
        self.text_ctrl_xtb_uhf.SetValue(str(config.DEFAULT_XTB_UHF))
        self.radio_box_xtb_solvation.SetStringSelection(str(config.DEFAULT_XTB_SOLVATION))
        self.choice_xtb_solvent.SetStringSelection(str(config.DEFAULT_XTB_SOLVENT))
        self.text_ctrl_xtb_cpu.SetValue(str(config.DEFAULT_XTB_CPU))
        self.text_ctrl_xtb_memory.SetValue(config.DEFAULT_XTB_MEMORY)
        # G16 Settings
        self.text_ctrl_g16_cpu.SetValue(str(config.DEFAULT_G16_CPU))
        self.text_ctrl_g16_memory.SetValue(str(config.DEFAULT_G16_MEMORY))
        self.text_ctrl_g16_init_guess_keywords.SetValue(config.DEFAULT_G16_GUESS_ADDITIONAL_KEYWORDS)

    def create_project(self, project_file: Path) -> None:
        self.project = NEBProjectNonBlocking()
        self.load_default_settings()
        self.update_project()
        self.project.work_dir = project_file.parent
        self.project.project_name = project_file.stem
        self.update_forms()
        self.update_all()
        self.text_ctrl_calculation_log.SetValue('')

    def load_project_file(self, json_file: Path) -> None:
        # Save and terminate current project
        if self.project is not None:
            self.project.save_json()
            if self.project.check() == 1:
                self.project.terminate()

        # read file and update GUI
        self.project = NEBProjectNonBlocking(json_file)
        self.update_forms()
        self.update_all()
        self.text_ctrl_calculation_log.SetValue('')

    def save_project_file(self) -> None:
        self.update_project()
        if self.project is not None:
            self.project.save_json()

    def update_project(self) -> None:
        if self.project is None:
            return

        self.project.opt_init = self.checkbox_init_opt.GetValue()
        self.project.opt_final = self.checkbox_final_opt.GetValue()

        self.project.num_images = int(self.text_ctrl_num_images.GetValue().strip())
        self.project.interpolation_method = self.choice_interpolation_method.GetStringSelection()

        self.project.neb_method = self.choice_neb_method.GetStringSelection()
        self.project.neb_k = float(self.text_ctrl_neb_k.GetValue().strip())
        self.project.neb_climb = self.checkbox_neb_climb.GetValue()
        self.project.neb_optimizer = self.choice_neb_optimizer.GetStringSelection()
        self.project.neb_fmax = float(self.text_ctrl_neb_fmax.GetValue().strip())
        self.project.neb_steps = int(self.text_ctrl_neb_steps.GetValue().strip())
        self.project.neb_parallel = int(self.text_ctrl_neb_parallel.GetValue().strip())
        # notebook page: XTB=0, G16=1
        page = self.notebook_calculator.GetSelection()
        if page == 0:
            self.project.calculator_type = 'xtb'
        elif page == 1:
            self.project.calculator_type = 'g16'

        self.project.xtb_gfn = self.choice_xtb_method.GetStringSelection()
        self.project.xtb_charge = int(self.text_ctrl_xtb_charge.GetValue().strip())
        self.project.xtb_uhf = int(self.text_ctrl_xtb_uhf.GetValue().strip())

        solvation = self.radio_box_xtb_solvation.GetStringSelection()
        if solvation.lower() == 'none':
            solvation = None
        self.project.xtb_solvation = solvation

        solvent = self.choice_xtb_solvent.GetStringSelection()
        if solvent.lower().strip() in ['none', '']:
            solvent = None
        self.project.xtb_solvent = solvent

        self.project.xtb_cpu = int(self.text_ctrl_xtb_cpu.GetValue().strip())
        self.project.xtb_memory_per_cpu = self.text_ctrl_xtb_memory.GetValue().strip()
        self.project.g16_cpu = int(self.text_ctrl_g16_cpu.GetValue().strip())
        self.project.g16_memory = self.text_ctrl_g16_memory.GetValue().strip()
        self.project.g16_guess_additional_keywords = self.text_ctrl_g16_init_guess_keywords.GetValue().strip()

        if self.project.check() == 0:
            self.finish_calculation()

    def update_forms(self) -> None:
        if self.project is None:
            return
        self.checkbox_init_opt.SetValue(self.project.opt_init)
        self.checkbox_final_opt.SetValue(self.project.opt_final)

        self.text_ctrl_num_images.SetValue(str(self.project.num_images))
        self.choice_interpolation_method.SetStringSelection(self.project.interpolation_method)

        self.choice_neb_method.SetStringSelection(self.project.neb_method)
        self.text_ctrl_neb_k.SetValue(str(self.project.neb_k))
        self.checkbox_neb_climb.SetValue(self.project.neb_climb)
        self.choice_neb_optimizer.SetStringSelection(self.project.neb_optimizer)
        self.text_ctrl_neb_fmax.SetValue(str(self.project.neb_fmax))
        self.text_ctrl_neb_steps.SetValue(str(self.project.neb_steps))
        self.text_ctrl_neb_parallel.SetValue(str(self.project.neb_parallel))
        # notebook page: XTB=0, G16=1
        if self.project.calculator_type.lower() == 'xtb':
            self.notebook_calculator.SetSelection(0)
        elif self.project.calculator_type.lower() == 'g16':
            self.notebook_calculator.SetSelection(1)

        self.choice_xtb_method.SetStringSelection(self.project.xtb_gfn)
        self.text_ctrl_xtb_charge.SetValue(str(self.project.xtb_charge))
        self.text_ctrl_xtb_uhf.SetValue(str(self.project.xtb_uhf))
        self.radio_box_xtb_solvation.SetStringSelection(str(self.project.xtb_solvation))
        self.choice_xtb_solvent.SetStringSelection(str(self.project.xtb_solvent))
        self.text_ctrl_xtb_cpu.SetValue(str(self.project.xtb_cpu))
        self.text_ctrl_xtb_memory.SetValue(self.project.xtb_memory_per_cpu)

        self.text_ctrl_g16_cpu.SetValue(str(self.project.g16_cpu))
        self.text_ctrl_g16_memory.SetValue(self.project.g16_memory)
        self.text_ctrl_g16_init_guess_keywords.SetValue(self.project.g16_guess_additional_keywords)

    def update_results(self) -> None:
        if self.project is None:
            self.text_ctrl_project.SetValue('Please create or open a project file.')
            self.text_ctrl_init_result.SetValue('not loaded')
            self.text_ctrl_final_result.SetValue('not loaded')
            self.text_ctrl_interpolation_result.SetValue('not done')
            self.text_ctrl_g16_template.SetValue('not loaded')
            self.list_box_neb_result_files.Clear()
            return

        # Check each result file and show
        self.text_ctrl_project.SetValue(str(self.project.work_dir / (self.project.project_name + '.json')))

        if self.project.init_traj_file().exists():
            self.text_ctrl_init_result.SetValue(self.project.init_traj_file().name)
        else:
            self.text_ctrl_init_result.SetValue('not loaded')

        if self.project.final_traj_file().exists():
            self.text_ctrl_final_result.SetValue(self.project.final_traj_file().name)
        else:
            self.text_ctrl_final_result.SetValue('not loaded')

        if self.project.initial_path_traj_file().exists():
            self.text_ctrl_interpolation_result.SetValue(self.project.initial_path_traj_file().name)
        else:
            self.text_ctrl_interpolation_result.SetValue('not done')

        # Avoid update when no change made:
        traj_names = [traj.stem for traj in self.project.get_all_neb_traj_files()]
        names_in_box = [name.rstrip('.traj') for name in self.list_box_neb_result_files.GetItems()]
        if traj_names != names_in_box:
            self.list_box_neb_result_files.Clear()
            for traj in traj_names:
                self.list_box_neb_result_files.Append(traj)

        if (self.project.work_dir / config.G16_TEMPLATE_FILE_NAME).exists():
            self.text_ctrl_g16_template.SetValue(config.G16_TEMPLATE_FILE_NAME)
        else:
            self.text_ctrl_g16_template.SetValue('not loaded')

    def update_calculation_log(self) -> None:
        if self.project is None:
            self.text_ctrl_current_calculation.SetValue('')
            self.text_ctrl_calculation_log.SetValue('')
            return
        current_job_name = self.project.current_calculation_job_name()
        if current_job_name is None:
            self.text_ctrl_current_calculation.SetValue('')
        elif current_job_name == 'opt_init':
            self.text_ctrl_current_calculation.SetValue('OPT Initial')
        elif current_job_name == 'opt_final':
            self.text_ctrl_current_calculation.SetValue('OPT Final')
        elif current_job_name == 'sp_init':
            self.text_ctrl_current_calculation.SetValue('SP Initial (no log)')
        elif current_job_name == 'sp_final':
            self.text_ctrl_current_calculation.SetValue('SP Final (no log)')
        elif current_job_name == 'g16_init_guess':
            self.text_ctrl_current_calculation.SetValue('G16 Init. Guess (no log)')
        elif current_job_name.startswith('neb'):
            self.text_ctrl_current_calculation.SetValue('NEB Calculation (#. {:})'.format(current_job_name[3:]))

        log_file = self.project.current_calculation_log_file()
        if log_file is not None:
            if log_file.exists():
                with log_file.open(mode='r') as f:
                    self.text_ctrl_calculation_log.SetValue(f.read())
                    self.text_ctrl_calculation_log.ShowPosition(self.text_ctrl_calculation_log.GetLastPosition())

    def update_all(self, update_form=False):
        self.update_project()
        self.update_results()
        self.update_calculation_log()
        if update_form:
            self.update_forms()
        self.save_project_file()

    def get_completed_neb_result(self) -> Optional[NEBResult]:
        if self.project is None:
            self.logging('Please make a project file first.')
            return

        # Read NEB trajectory
        traj_name = self.list_box_neb_result_files.GetStringSelection()
        if traj_name:
            neb_number = int(traj_name.split('_')[-1])
        else:
            self.logging('No valid trajectory file selected.')
            return
        traj_file = self.project.neb_path_traj_file(neb_number)
        if not traj_file.exists():
            self.logging('No trajectory file.')
            return
        result = NEBResult(traj_file, num_nodes=self.project.num_images+2)

        # Read init (for energy of node 0)
        traj_file = self.project.init_traj_file()
        if not traj_file.exists():
            self.logging('No init trajectory file (required for energy).')
            return
        traj = SingleTrajectory(traj_file)
        init_energy = traj.energies[-1]
        if init_energy is np.nan:
            self.logging('Init trajectory file lacks energy information.')
            return

        # Read final (for energy of node -1)
        traj_file = self.project.final_traj_file()
        if not traj_file.exists():
            self.logging('No final trajectory file (required for energy).')
            return
        traj = SingleTrajectory(traj_file)
        final_energy = traj.energies[-1]
        if final_energy is np.nan:
            self.logging('Final trajectory file lacks energy information.')
            return

        result.complete_energy(0, init_energy)
        result.complete_energy(-1, final_energy)
        if not result.is_energy_completed():
            self.logging('Energy information is not completed. This is quite unusual.')
            return

        return result

    def on_timer(self, event) -> None:
        self.update_all()

    def on_close(self, event) -> None:
        if self.timer.IsRunning():
            self.timer.Stop()

        if self.calc_all_thread is not None:
            dialog = wx.MessageDialog(self.frame,
                                      "Do you want to terminate current calculation and exit?",
                                      "Calculation is Running",
                                      wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION)
            result = dialog.ShowModal()
            if result == wx.ID_YES:
                self.calc_all_thread.terminate()
                self.update_all()
                self.save_project_file()
                self.frame.Destroy()
            else:
                self.timer.Start(config.CHECK_INTERVAL)
                event.Veto()

        elif self.project is not None:
            if self.project.check() == 1:
                dialog = wx.MessageDialog(self.frame,
                                          "Do you want to terminate current calculation and exit?",
                                          "Calculation is Running",
                                          wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION)
                result = dialog.ShowModal()
                if result == wx.ID_YES:
                    self.project.terminate()
                    self.update_all()
                    self.save_project_file()
                    self.frame.Destroy()
                else:
                    self.timer.Start(config.CHECK_INTERVAL)
                    event.Veto()
            else:
                self.update_all()
                self.save_project_file()
                self.frame.Destroy()

        else:
            self.frame.Destroy()

    @with_timer_stop
    def on_button_project_new(self, event):
        if self.project is not None:
            if self.project.check() == 1:
                self.logging('Calculation is running. Please stop it before creating a new project.')
                return
        dialog = wx.FileDialog(None, 'New project file (JSON)',
                               wildcard='JSON file (*.json)|*.json|All files (*.*)|*.*',
                               style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dialog.ShowModal() == wx.ID_OK:
            file = Path(dialog.GetPath())
            self.create_project(file)
        dialog.Destroy()

    @with_timer_stop
    def on_button_project_open(self, event):
        if self.project is not None:
            if self.project.check() == 1:
                self.logging('Calculation is running. Please stop it before creating a new project.')
                return
        dialog = wx.FileDialog(None, 'New project file (JSON)',
                               wildcard='JSON file (*.json)|*.json|All files (*.*)|*.*',
                               style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_OK:
            file = Path(dialog.GetPath())
            self.load_project_file(file)
        dialog.Destroy()

    @with_timer_stop
    def on_button_init_structure_open(self, event) -> None:
        dialog = wx.FileDialog(None, 'Structure file',
                               wildcard='All files (*.*)|*.*',
                               style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_OK:
            file = Path(dialog.GetPath())
            self.text_ctrl_init_structure_file.SetValue(str(file))
        dialog.Destroy()

    @with_timer_stop
    def on_button_init_load(self, event) -> None:
        if self.project is None:
            self.logging('Please make a project file first.')
            return
        if self.project.check() == 1:
            self.logging('Calculation is now running. Please wait or stop the calculation.')
            return
        if self.calc_all_thread is not None:
            self.logging('Calculation is now running. Please wait or stop the calculation.')
            return

        structure_file = Path(self.text_ctrl_init_structure_file.GetValue()).absolute()

        if not structure_file.is_file():
            self.logging('Structure file not found.')
            return

        self.update_project()
        self.project.load_init_structure(structure_file)
        self.enable_calculation_buttons(False)
        self.update_all()

    @with_timer_stop
    def on_final_structure_open(self, event) -> None:
        dialog = wx.FileDialog(None, 'Structure file',
                               wildcard='All files (*.*)|*.*',
                               style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_OK:
            file = Path(dialog.GetPath())
            self.text_ctrl_final_structure_file.SetValue(str(file))
        dialog.Destroy()

    @with_timer_stop
    def on_button_final_load(self, event) -> None:
        if self.project is None:
            self.logging('Please make a project file first.')
            return
        if self.project.check() == 1:
            self.logging('Calculation is now running. Please wait or stop the calculation.')
            return
        if self.calc_all_thread is not None:
            self.logging('Calculation is now running. Please wait or stop the calculation.')
            return

        structure_file = Path(self.text_ctrl_final_structure_file.GetValue()).absolute()

        if not structure_file.is_file():
            self.logging('Structure file not found.')
            return

        self.update_project()
        self.project.load_final_structure(structure_file)
        self.enable_calculation_buttons(False)
        self.update_all()

    @with_timer_stop
    def on_button_interpolation_run(self, event) -> None:
        if self.project is None:
            self.logging('Please make a project file first.')
            return
        if self.project.check() == 1:
            self.logging('Calculation is now running. Please wait or stop the calculation.')
            return
        if self.calc_all_thread is not None:
            self.logging('Calculation is now running. Please wait or stop the calculation.')
            return

        self.update_project()
        self.project.interpolate()  # Currently, this is a blocking process.
        self.update_all()

    @with_timer_stop
    def on_button_neb_run(self, event) -> None:
        if self.project is None:
            self.logging('Please make a project file first.')
            return
        if self.project.check() == 1:
            self.logging('Calculation is now running. Please wait or stop the calculation.')
            return
        if self.calc_all_thread is not None:
            self.logging('Calculation is now running. Please wait or stop the calculation.')
            return

        self.update_project()
        self.project.run_neb()
        self.enable_calculation_buttons(False)
        self.update_all()

    @with_timer_stop
    def on_button_g16_template_load(self, event) -> None:
        if self.project is None:
            self.logging('Please make a project file first.')
            return
        wildcard = 'Gaussian Input Files (*.gjf;*.gjc;*.com)|*.gjf;*.gjc;*.com|All files (*.*)|*.*'
        dialog = wx.FileDialog(None, 'Gaussian Input template file',
                               wildcard=wildcard,
                               style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_OK:
            file = Path(dialog.GetPath())
            shutil.copy(file, self.project.work_dir / config.G16_TEMPLATE_FILE_NAME)
        dialog.Destroy()
        self.update_all()

    @with_timer_stop
    def on_button_g16_template_edit(self, event) -> None:
        if self.project is None:
            self.logging('Please make a project file first.')
            return
        template = self.project.work_dir / config.G16_TEMPLATE_FILE_NAME
        if not template.exists():
            with template.open(mode='w', encoding='utf-8', newline='\n') as f:
                f.writelines([
                    '#P M06L def2SVP densityfit scf=xqc\n',
                    '\n',
                    'Just input keywords without link0 lines, job-type, sym, nosym, guess. \n',
                    '\n',
                    '0 1\n',
                    '@\n',
                    '\n'
                ])
        sp.Popen([config.TEXT_EDITOR_PATH, str(template)])
        self.update_all()

    @with_timer_stop
    def on_button_g16_init_guess_run(self, event) -> None:
        if self.project is None:
            self.logging('Please make a project file first.')
            return
        if self.project.check() == 1:
            self.logging('Calculation is now running. Please wait or stop the calculation.')
            return
        if self.calc_all_thread is not None:
            self.logging('Calculation is now running. Please wait or stop the calculation.')
            return

        self.update_project()
        self.project.run_g16_init_guess()
        self.enable_calculation_buttons(False)
        self.update_all()

    def on_button_init_result_view(self, event) -> None:
        if self.project is None:
            self.logging('Please make a project file first.')
            return

        traj_file = self.project.init_traj_file()
        xyz_file = self.project.init_xyz_file()

        if not traj_file.exists():
            self.logging('No trajectory file.')
            return
        if not xyz_file.exists():
            traj = SingleTrajectory(traj_file)
            traj.save_xyz(xyz_file, energy_unit='eV')
        sp.Popen([config.VIEWER_PATH, str(xyz_file)])

    def on_button_final_result_view(self, event) -> None:
        if self.project is None:
            self.logging('Please make a project file first.')
            return
        traj_file = self.project.final_traj_file()
        xyz_file = self.project.final_xyz_file()

        if not traj_file.exists():
            self.logging('No trajectory file.')
            return
        if not xyz_file.exists():
            traj = SingleTrajectory(traj_file)
            traj.save_xyz(xyz_file, energy_unit='eV')
        sp.Popen([config.VIEWER_PATH, str(xyz_file)])

    def on_button_interpolation_result_view(self, event) -> None:
        if self.project is None:
            self.logging('Please make a project file first.')
            return

        traj_file = self.project.initial_path_traj_file()
        xyz_file = self.project.initial_path_xyz_file()

        if not traj_file.exists():
            self.logging('No trajectory file.')
            return
        if not xyz_file.exists():
            traj = SingleTrajectory(traj_file)
            traj.save_xyz(xyz_file, energy_unit='eV')
        sp.Popen([config.VIEWER_PATH, str(xyz_file)])

    def on_button_neb_view(self, event) -> None:
        if self.project is None:
            self.logging('Please make a project file first.')
            return
        traj_name = self.list_box_neb_result_files.GetStringSelection()
        if traj_name:
            neb_number = int(traj_name.split('_')[-1])
        else:
            self.logging('No valid trajectory file selected.')
            return

        traj_file = self.project.neb_path_traj_file(neb_number)
        xyz_file = self.project.neb_path_optimized_xyz_file(neb_number)

        if not traj_file.exists():
            self.logging('No trajectory file.')
            return
        if not xyz_file.exists():
            traj = NEBResult(traj_file, num_nodes=self.project.num_images+2)
            traj.save_xyz(xyz_file, iteration=-1, energy_unit='eV')
        sp.Popen([config.VIEWER_PATH, str(xyz_file)])

    def on_button_neb_view_all(self, event) -> None:
        if self.project is None:
            self.logging('Please make a project file first.')
            return
        traj_name = self.list_box_neb_result_files.GetStringSelection()
        if traj_name:
            neb_number = int(traj_name.split('_')[-1])
        else:
            self.logging('No valid trajectory file selected.')
            return

        traj_file = self.project.neb_path_traj_file(neb_number)
        xyz_file = self.project.neb_path_xyz_file(neb_number)
        if not traj_file.exists():
            self.logging('No trajectory file.')
            return
        if not xyz_file.exists():
            traj = NEBResult(traj_file, num_nodes=self.project.num_images+2)
            traj.save_xyz(xyz_file, energy_unit='eV')
        sp.Popen([config.VIEWER_PATH, str(xyz_file)])

    def on_button_neb_plot(self, event) -> None:
        result = self.get_completed_neb_result()
        if result is None:
            return
        result.plot_mep(iteration=-1, energy_unit='kcal/mol')

    def on_button_neb_plot_all(self, event) -> None:
        result = self.get_completed_neb_result()
        if result is None:
            return
        result.plot_mep_all(energy_unit='kcal/mol')

    def on_button_neb_info(self, event) -> None:
        result = self.get_completed_neb_result()
        if result is None:
            return
        text = 'EA (kcal/mol) | # of Peak Top | RE (kcal/mol)\n'
        text += '{:>13.4f} | {:>13d} | {:>13.4f}\n'.format(
            result.get_barrier(iteration=-1, energy_unit='kcal/mol'),
            result.get_highest_node_index(iteration=-1),
            result.get_reaction_energy_change(iteration=-1, energy_unit='kcal/mol')
        )

        text_view = TextViewFrame(self.frame, 'NEB Results', text)
        text_view.Show(True)

    def on_button_neb_info_all(self, event) -> None:
        result = self.get_completed_neb_result()
        if result is None:
            return
        text = 'Iteration | EA (kcal/mol) | # of Peak Top | RE (kcal/mol)\n'
        for i in range(result.num_iteration):
            text += '{:>9d} | {:>13.4f} | {:>13d} | {:>13.4f}\n'.format(
                i,
                result.get_barrier(iteration=i, energy_unit='kcal/mol'),
                result.get_highest_node_index(iteration=i),
                result.get_reaction_energy_change(iteration=i, energy_unit='kcal/mol')
            )
        text_view = TextViewFrame(self.frame, 'NEB Results', text)
        text_view.Show(True)

    def on_button_neb_save_ts(self, event) -> None:
        result = self.get_completed_neb_result()
        if result is None:
            return
        dialog = wx.FileDialog(None, 'Save XYZ file',
                               wildcard='XYZ file (*.xyz)|*.xyz|All files (*.*)|*.*',
                               style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dialog.ShowModal() == wx.ID_OK:
            file = Path(dialog.GetPath())
            result.save_xyz(xyz_file=file,
                            iteration=-1,
                            node=result.get_highest_node_index(),
                            energy_unit='eV')
        dialog.Destroy()

    def on_button_neb_delete(self, event) -> None:
        if self.project is None:
            self.logging('Please make a project file first.')
            return
        traj_name = self.list_box_neb_result_files.GetStringSelection()
        if traj_name:
            neb_number = int(traj_name.split('_')[-1])
        else:
            self.logging('No valid trajectory file selected.')
            return

        delete_files = [
            self.project.neb_path_traj_file(neb_number),
            self.project.neb_path_xyz_file(neb_number),
            self.project.neb_path_optimized_xyz_file(neb_number)
        ]
        for file in delete_files:
            remove(file)

        self.update_all()

    @with_timer_stop
    def on_button_stop(self, event) -> None:
        if self.project is None:
            self.enable_calculation_buttons(True)
            self.update_all()
            return

        if self.calc_all_thread is not None:
            self.calc_all_thread.terminate()
            self.update_all()
            return

        if self.project.check() == 1:
            self.project.terminate()
            self.logging('Calculation has been terminated.')
            self.enable_calculation_buttons(True)
            self.update_all()
            return

    def on_button_log_delete(self, event) -> None:
        self.text_ctrl_log.SetValue('')

    def on_button_run_all(self, event) -> None:

        if self.project is None:
            self.logging('Please make a project file first.')
            return
        if self.project.check() == 1:
            self.logging('Calculation is now running. Please wait or stop the calculation.')
            return
        if self.calc_all_thread is not None:
            self.logging('Calculation is now running. Please wait or stop the calculation.')
            return

        self.project.clear_all_results()

        # check init and final files
        init_structure_file = Path(self.text_ctrl_init_structure_file.GetValue()).absolute()
        if not init_structure_file.is_file():
            self.logging('Init structure file not found.')
            return
        final_structure_file = Path(self.text_ctrl_final_structure_file.GetValue()).absolute()
        if not final_structure_file.is_file():
            self.logging('Init structure file not found.')
            return

        self.timer.Stop()
        self.update_project()
        self.enable_calculation_buttons(False)

        self.calc_all_thread = CalcAllThread(init_structure_file=init_structure_file,
                                             final_structure_file=final_structure_file,
                                             neb_project=self.project,
                                             parent_window=self)
        self.logging('All calculations will be run.')

    def on_all_calc_end_event(self, event: AllCalcEndEvent):
        if event.terminated:
            wx.MessageBox('Calculation has been terminated', 'Calculation Terminated', wx.OK | wx.ICON_INFORMATION)
        else:
            wx.MessageBox('Calculation Finished', 'Calculation Finished', wx.OK | wx.ICON_INFORMATION)
        self.calc_all_thread = None
        self.update_all()
        self.timer.Start(config.CHECK_INTERVAL)

    def on_emergency_reset(self, event):
        """
        Emergency function to stop all calculation and reset timer and etc.
        """
        if self.calc_all_thread is not None:
            self.calc_all_thread.terminate()
            self.calc_all_thread = None
        if self.project is not None:
            if self.project.check() == 1:
                self.project.terminate()

        self.update_all()
        self.update_forms()
        self.enable_calculation_buttons(True)

        if self.timer.IsRunning():
            self.timer.Start(config.CHECK_INTERVAL)

        self.logging('Emergency reset.')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # CPU and Memory Settings for main ASE process
    if config.ASE_OMP_NUM_THREADS is not None:
        omp_num_threads = str(config.ASE_OMP_NUM_THREADS)
    else:
        from multiprocessing import cpu_count
        omp_num_threads = str(cpu_count())
    os.environ['OMP_NUM_THREADS'] = str(config.ASE_OMP_NUM_THREADS)
    os.environ['KMP_NUM_THREADS'] = str(config.ASE_OMP_NUM_THREADS)
    if config.ASE_OMP_STACKSIZE is not None:
        os.environ['OMP_STACKSIZE'] = config.ASE_OMP_STACKSIZE.rstrip('b').rstrip('B').rstrip('w').rstrip('b')

    app = XTBNEBApp(True)
    app.MainLoop()
