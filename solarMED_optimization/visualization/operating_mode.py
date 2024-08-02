from enum import Enum
from lxml import etree
from pydantic import BaseModel, Field, ConfigDict
from phd_visualizations.diagrams import nsmap
from loguru import logger
from copy import deepcopy
from typing import Literal

logger.disable("operating_mode")

class SolarFieldState(Enum):
    IDLE = 0
    ACTIVE = 1


class ThermalStorageState(Enum):
    IDLE = 0
    ACTIVE = 1


class MEDState(Enum):
    OFF = 0
    GENERATING_VACUUM = 1
    STARTING_UP = 2
    SHUTTING_DOWN = 3
    IDLE = 4
    ACTIVE = 5


color_palette: dict = {
    'gray': '#E6E6E6',
    'red': '#B85450',
    'blue': '#6C8EBF',
    'yellow': '#FFF2CC',
    'green': '#97D077'

}


def find_object(object_id: str, diagram: etree._ElementTree, group: bool = False) -> etree.Element:
    if not group:
        object = diagram.xpath(f'//svg:g[@id="cell-{object_id}"]', namespaces=nsmap)
    else:
        object = diagram.xpath(f'//svg:g[starts-with(@id, "cell-{object_id}")]', namespaces=nsmap)

    if not object:
        raise ValueError(f'Object {object_id} not found in diagram')
    else:
        return object


def change_line_color(object_id: str, diagram: etree._ElementTree, color: str, not_inplace: bool = False,
                      tag_key: Literal['rect', 'path'] = 'path', group: bool = False,
                      stop_on_first_change=False) -> etree.Element or None:
    centinela = False

    def change_property(child):

        if tag_key in child.tag:
            logger.debug(f'Changing color of {object_id}/{child.get("id")} to {color}')
            child.set('stroke', str(color))

            nonlocal centinela
            centinela = True

            if stop_on_first_change:
                return

        elif len(child) > 0:
            for child_ in child:
                change_property(child_)

    if not_inplace:
        diagram = deepcopy(diagram)

    object = find_object(object_id, diagram, group=group)

    for child in object:
        # if centinela and stop_on_first_change:
        #     break

        change_property(child)

    if not centinela:
        logger.error(f'Could not find any {tag_key} object to change its width in {object_id}')

    return diagram if not_inplace else None

def change_bg_color(object: etree._Element, color: str, not_inplace: bool = False,
                    tag_key: Literal['rect', 'path'] = 'rect') -> etree.Element or None:
    centinela = False

    def change_color(child):

        if tag_key in child.tag:
            logger.debug(f'Changing fill color of {child.get("id")} to {color}')
            child.set('fill', color)

            nonlocal centinela
            centinela = True

            return

        elif len(child) > 0:
            for child_ in child:
                change_color(child_)

    if not_inplace:
        object = deepcopy(object)

    for child in object:
        change_color(child)

    # if not centinela:
    #     logger.error(f'Could not find any path object to change its fill color in {object_id}')

    return object if not_inplace else None


def change_line_width(object_id: str, diagram: etree._ElementTree, width: int, not_inplace: bool = False,
                      tag_key: Literal['rect', 'path'] = 'path', group: bool = False,
                      stop_on_first_change=False) -> etree.Element or None:
    centinela = False

    def change_width(child):

        if tag_key in child.tag:
            logger.debug(f'Changing width of {object_id}/{child.get("id")} to {width}')
            child.set('stroke-width', str(width))

            nonlocal centinela
            centinela = True

            if stop_on_first_change:
                return

        elif len(child) > 0:
            for child_ in child:
                change_width(child_)

    if not_inplace:
        diagram = deepcopy(diagram)

    object = find_object(object_id, diagram, group=group)

    for child in object:
        # if centinela and stop_on_first_change:
        #     break

        change_width(child)

    if not centinela:
        logger.error(f'Could not find any {tag_key} object to change its width in {object_id}')

    return diagram if not_inplace else None

class SolarMEDState(BaseModel):
    sf_state: SolarFieldState
    ts_state: ThermalStorageState
    med_state: MEDState

    sf_bg_colors: dict = {
        SolarFieldState.IDLE: color_palette['gray'],
        SolarFieldState.ACTIVE: color_palette['green']
    }

    ts_bg_colors: dict = {
        ThermalStorageState.IDLE: color_palette['gray'],
        ThermalStorageState.ACTIVE: color_palette['green']
    }

    med_bg_colors: dict = {
        MEDState.OFF: '#f0f0f0',
        MEDState.STARTING_UP: color_palette['yellow'],
        MEDState.SHUTTING_DOWN: color_palette['yellow'],
        MEDState.IDLE: color_palette['yellow'],
        MEDState.ACTIVE: color_palette['green'],
        MEDState.GENERATING_VACUUM: color_palette['yellow']
    }

    max_line_width: int = Field(25, gt=0)
    min_line_width: int = Field(3, gt=0)

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    def model_post_init(self, ctx):
        # self._state = (self.solar_field, self.thermal_storage, self.med)
        self._mean_line_width = round((self.max_line_width + self.min_line_width) / 2, 2)

        # Set background colors
        self._med_bg_color = self.med_bg_colors[self.med_state]
        self._ts_bg_color = self.ts_bg_colors[self.ts_state]
        self._sf_bg_color = self.sf_bg_colors[self.sf_state]

        # Solar field
        self._sf_line_width = self._mean_line_width if self.sf_state == SolarFieldState.ACTIVE else self.min_line_width
        self._ts_line_width = self._mean_line_width if self.ts_state == ThermalStorageState.ACTIVE else self.min_line_width

        self._sf_line_color = None
        self._sf_in_line_color = None
        if self.sf_state == SolarFieldState.IDLE:
            self._sf_line_color = color_palette['gray']
        elif self.sf_state == SolarFieldState.ACTIVE:
            if self.ts_state == ThermalStorageState.IDLE:
                self._sf_line_color = color_palette['red']
            else:
                # Should be de default color, but for some reason it is not working
                self._sf_in_line_color = color_palette['blue']

        # Thermal storage
        self._ts_h_in_line_color = color_palette['red'] if self.sf_state == SolarFieldState.ACTIVE else color_palette[
            'blue']
        self._ts_line_color = color_palette['gray'] if self.ts_state == ThermalStorageState.IDLE else None

        # MED
        self._med_line_width = self.min_line_width
        self._med_vacuum_line_width = self.min_line_width
        self._med_line_color = None
        self._med_vacuum_color = None

        if self.med_state == MEDState.ACTIVE or self.med_state == self.med_state.STARTING_UP:
            self._med_line_width = self._mean_line_width
            self._med_vacuum_line_width = self._mean_line_width

        elif self.med_state == MEDState.SHUTTING_DOWN:
            self._med_line_width = self.min_line_width
            self._med_vacuum_line_width = self.min_line_width
            self._med_vacuum_color = color_palette['gray']

        elif self.med_state == self.med_state.IDLE:
            self._med_vacuum_line_width = self._mean_line_width
            self._med_line_color = color_palette['gray']
            self._med_vacuum_color = color_palette['blue']

        elif self.med_state == MEDState.GENERATING_VACUUM:
            self._med_vacuum_line_width = self.max_line_width
            self._med_line_color = color_palette['gray']
            self._med_vacuum_color = color_palette['blue']

        elif self.med_state == MEDState.OFF:
            self._med_line_color = color_palette['gray']
            self._med_vacuum_color = color_palette['gray']

        # else:
        #     raise ValueError(f'Unknown MED state: {self.med_state}')

    def change_bg_colors(self, src_diagram: etree.ElementTree) -> etree.ElementTree:
        for id, object_id, tag_key in zip(['sf', 'ts', 'med'], ["bg_sf", ["bg_ts", "bg_hx"], "bg_med"],
                                          ['rect', ['path', 'rect'], 'rect']):

            if not isinstance(object_id, list):
                object_id = [object_id]
                tag_key = [tag_key]

            for object_id_, tag_key_ in zip(object_id, tag_key):
                object = find_object(object_id_, src_diagram)
                change_bg_color(object, getattr(self, f'_{id}_bg_color'), tag_key=tag_key_)

        return src_diagram

    def create_state_diagram(self, src_diagram: etree.ElementTree) -> etree.ElementTree:
        # Change background colors
        self.change_bg_colors(src_diagram)

        # Change line widths
        src_diagram = change_line_width('line_med', diagram=src_diagram, width=self._med_line_width, group=True,
                                        not_inplace=True)
        src_diagram = change_line_width('line_med_vacuum', diagram=src_diagram, width=self._med_vacuum_line_width,
                                        group=True, not_inplace=True)
        src_diagram = change_line_width('line_sf', diagram=src_diagram, width=self._sf_line_width, group=True,
                                        not_inplace=True)
        src_diagram = change_line_width('line_ts', diagram=src_diagram, width=self._ts_line_width, group=True,
                                        not_inplace=True)

        # Change line colors
        if self._sf_line_color:
            src_diagram = change_line_color('line_sf', diagram=src_diagram, color=self._sf_line_color, group=True,
                                            not_inplace=True)
        if self._sf_in_line_color:
            src_diagram = change_line_color('line_sf_in', diagram=src_diagram, color=self._sf_in_line_color, group=True,
                                            not_inplace=True)
        if self._sf_line_color:
            src_diagram = change_line_color('line_sf', diagram=src_diagram, color=self._sf_line_color, group=True,
                                            not_inplace=True)
        src_diagram = change_line_color('line_ts_h_in', diagram=src_diagram, color=self._ts_h_in_line_color, group=True,
                                        not_inplace=True)
        if self._med_line_color:
            src_diagram = change_line_color('line_med', diagram=src_diagram, color=self._med_line_color, group=True,
                                            not_inplace=True)
        if self._med_vacuum_color:
            src_diagram = change_line_color('line_med_vacuum', diagram=src_diagram, color=self._med_vacuum_color,
                                            group=True, not_inplace=True)

        if self._ts_line_color:
            src_diagram = change_line_color('line_ts', diagram=src_diagram, color=self._ts_line_color, group=True,
                                            not_inplace=True)

        return src_diagram