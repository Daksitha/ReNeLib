var mousePosition, active_element;
var offset = [0, 0];
var isDown = false;
var slider_min = -11;

var default_range = [0, 100];
var default_round = 0;
var default_color = '#7ef4ff';

var slider_positions = {};
var slider_percentages = {};
var slider_values = {};

const posesCharamel = ["brows_up", "brows_down", "brows_tight_up", "brows_tight_down", "lids_upper_rotated_closing", "lids_wide_open", "lids_lower_up", "lids_lower_down", "lids_upper_up", "lids_upper_down", "nose_up", "nose_down", "nosewings_wide", "nosewings_tight", "cheeks_up", "cheeks_down", "mouth_up", "mouth_down", "mouth_open", "mouth_tight", "mouth_wide", "mouth_rotate_in", "mouth_rotate_out", "jaw_front", "jaw_back", "jaw_down", "tongue_tip_gum_up", "tongue_tip_gum_down", "tongue_tip_teeth_upper", "tongue_tip_teeth_lower", "tongue_tip_stick_out", "phon_a", "phon_b", "phon_d", "phon_e", "phon_f", "phon_o", "phon_sch", "mimic_blink", "mimic_kiss", "mimic_winkl", "mimic_lidl_close", "mimic_lidr_close", "mimic_brows_down", "mimic_brows_down_right", "mimic_brows_down_left", "mimic_brows_up", "mimic_brows_up_right", "mimic_brows_up_left", "mimic_blow", "mimic_breath", "emot_demanding", "emot_surprised", "emot_happy", "emot_smile", "emot_angry", "emot_crazy", "emot_disgust", "emot_pensively", "emot_bored", "emot_sad", "emot_disappointed", "browouterupleft", "browouterupright", "browinnerup", "browdownleft", "browdownright", "eyeblinkleft", "eyeblinkright", "eyelookinleft", "eyelookoutleft", "eyelookupleft", "eyelookdownleft", "eyelookinright", "eyelookoutright", "eyelookupright", "eyelookdownright", "eyesquintleft", "eyesquintright", "eyewideleft", "eyewideright", "cheekpuff", "cheeksquintleft", "cheeksquintright", "nosesneerleft", "nosesneerright", "jawforward", "jawopen", "jawleft", "jawright", "mouthclose", "mouthfunnel", "mouthdimpleleft", "mouthdimpleright", "mouthfrownleft", "mouthfrownright", "mouthsmileleft", "mouthsmileright", "mouthpucker", "mouthlowerdownleft", "mouthlowerdownright", "mouthleft", "mouthright", "tongueout", "mouthpressleft", "mouthpressright", "mouthrollupper", "mouthrolllower", "mouthstretchleft", "mouthstretchright", "mouthshruglower", "mouthshrugupper", "mouthupperupleft", "mouthupperupright"]



document.addEventListener('mouseup', function() {
  isDown = false;
  document.body.style.webkitUserSelect = '';
  document.body.style.mozUserSelect = '';
  document.body.style.msUserSelect = '';
}, true);

document.addEventListener('mousemove', function(event) {
  if (isDown) {
    mousePosition = {
      x: event.clientX,
      y: event.clientY
    };

    var current_input = active_element.parentElement.parentElement.parentElement.childNodes[1];
    var slider_groover = active_element.parentElement.firstChild;
    var name = current_input.name;
    var slider_max = slider_groover.clientWidth + slider_min;
    var min = parseFloat(current_input.min);
    var max = parseFloat(current_input.max);
    if ((min !== '' && max !== '') && (min < max)) {
      var range = [min, max]
    } else {
      var range = default_range;
    }
    var left_pos = mousePosition.x + offset[0];

    if (left_pos < slider_min) {
      slider_positions[name] = slider_min;
    } else if (left_pos > slider_max) {
      slider_positions[name] = slider_max + 2;
    } else {
      slider_positions[name] = left_pos;
    }

    var percentages = 100 * (slider_positions[name] - slider_min - 2) / slider_groover.clientWidth;
    var value = range[0] + (range[1] - range[0]) * percentages / 100;

    setSliderTo(name, value);
  }
}, true);

var sliders = document.getElementsByClassName('slider1');

for (var i = 0; i < sliders.length; i++) {
  var slider_parent = createSuperElement('div', {
    'class': 'slider1_parent'
  });
  sliders[i].parentNode.insertBefore(slider_parent, sliders[i]);
  slider_parent.appendChild(sliders[i]);

  if (sliders[i].getAttribute('text')) {
    var text = createSuperElement('p', {
      'class': 'title'
    }, sliders[i].getAttribute('text'));
  } else {
    var text = createSuperElement('span');
  }

  slider_parent.insertBefore(text, sliders[i]);

  var color = sliders[i].getAttribute('color') !== null ? sliders[i].getAttribute('color') : default_color;

  var slider_main_block = createSuperElement('div', {
    'class': 'main_block'
  });
  var slider_groove_parent = createSuperElement('div', {
    'class': 'groove_parent'
  });
  var slider_groove = createSuperElement('div', {
    'class': 'groove'
  });
  var slider_fill = createSuperElement('div', {
    'class': 'fill'
  }, '', {
    'background-color': color
  });
  var slider_rider = createSuperElement('div', {
    'class': 'rider'
  });

  var min = parseFloat(sliders[i].min);
  var max = parseFloat(sliders[i].max);
  if ((min !== '' && max !== '') && (min < max)) {
    var range = [min, max]
  } else {
    var range = default_range;
  }

  var table_data = [
    [
      [range[0], {
        'class': 'left'
      }],
      [range[1], {
        'class': 'right'
      }]
    ]
  ];
  var slider_range = createSuperTable(table_data, {
    'class': 'slider_range'
  });

  slider_groove.appendChild(slider_fill);
  slider_groove_parent.appendChild(slider_groove);
  slider_groove_parent.appendChild(slider_rider);
  slider_main_block.appendChild(slider_groove_parent);
  slider_main_block.appendChild(slider_range);
  slider_parent.appendChild(slider_main_block);

  slider_rider.addEventListener('mousedown', function(e) {
    var current_input = this.parentElement.parentElement.parentElement.childNodes[1];

    isDown = true;
    offset[0] = this.offsetLeft - e.clientX;
    active_element = this;

    if (current_input.getAttribute('animate') !== 'no') {
      this.parentNode.lastChild.style.transition = '';
      this.parentNode.firstChild.firstChild.style.transition = '';
    }

    document.body.style.webkitUserSelect = 'none';
    document.body.style.mozUserSelect = 'none';
    document.body.style.msUserSelect = 'none';

  }, true);

  slider_groove.addEventListener('click', function(e) {
    var current_input = this.parentElement.parentElement.parentElement.childNodes[1];
    var name = current_input.name;
    var click_position = e.clientX - my_offset(this).left;

    var min = parseFloat(current_input.min);
    var max = parseFloat(current_input.max);
    if ((min !== '' && max !== '') && (min < max)) {
      var range = [min, max]
    } else {
      var range = default_range;
    }

    if (current_input.getAttribute('animate') !== 'no') {
      this.parentNode.lastChild.style.transition = 'left 0.2s ease-in-out';
      this.parentNode.firstChild.firstChild.style.transition = 'width 0.2s ease-in-out';
    }

    var percentages = 100 * (click_position) / (this.clientWidth + 2);
    var value = range[0] + (range[1] - range[0]) * percentages / 100;
    setSliderTo(name, value);

  }, true);

  sliders[i].addEventListener('change', function(e) {
    setSliderTo(this.name, this.value);
  }, true);

  if (!sliders[i].value) sliders[i].value = 0;
  setSliderTo(sliders[i].name, sliders[i].value);
}

function setSliderTo(name, value) {
  var slider = document.getElementsByName(name)[0];
  value = parseFloat(value);

  var min = parseFloat(slider.min);
  var max = parseFloat(slider.max);
  if ((min !== '' && max !== '') && (min < max)) {
    var range = [min, max]
  } else {
    var range = default_range;
  }

  if (value >= range[0] && value <= range[1] && !isNaN(value)) {
    var data_round = slider.getAttribute('round') !== null ? slider.getAttribute('round') : default_round;
    if (slider.getAttribute('smooth') !== 'yes') value = round(value, data_round);
    slider_percentages[name] = 100 * (value - range[0]) / (range[1] - range[0]);
    slider.parentNode.childNodes[2].firstChild.firstChild.firstChild.style.width = round(slider_percentages[name], 2) + '%';
    slider.parentNode.childNodes[2].firstChild.lastChild.style.left = 'calc(' + round(slider_percentages[name], 2) + '% - 11px )';
    value = round(value, data_round);
    slider.value = value;
    slider_values[name] = value;

  } else {
    //console.log('Value ['+value+'] is out of slider range: '+range[0]+'-'+range[1] || default_range[1]);
    if (value < range[0] && !isNaN(value)) setSliderTo(name, range[0]);
    else if (value > range[1] && !isNaN(value)) setSliderTo(name, range[1]);
    else slider.value = slider_values[name];
  }

  try {
    slider.onchange(vm.models.getFirst().setPoseByName(name, (slider.value)));
    slider.onclick(vm.models.getFirst().setPoseByName(name, (slider.value)));
  } catch (err) {}
}

function my_offset(elem) {
  if (!elem) elem = this;

  var x = elem.offsetLeft;
  var y = elem.offsetTop;

  while (elem == elem.offsetParent) {
    x += elem.offsetLeft;
    y += elem.offsetTop;
  }

  return {
    left: x,
    top: y
  };
}

function round(value, precision) {
  var multiplier = Math.pow(10, precision || 0);
  return Math.round(value * multiplier) / multiplier;
}

var element;
var table;

function createSuperElement(type, attributes, innerHTML, style) {
  if (attributes === undefined) attributes = '';
  if (innerHTML === undefined) innerHTML = '';
  if (style === undefined) style = '';

  element = document.createElement(type);

  if (innerHTML !== '' && typeof innerHTML === 'object') element.appendChild(innerHTML);
  else if (innerHTML !== '') element.innerHTML = innerHTML;

  if (attributes !== '') {
    for (var i in attributes) {
      element.setAttribute(i, attributes[i]);
    }
  }

  if (style !== '') {
    var styles = '';
    for (var i in style) {
      styles += i + ':' + style[i] + ';';
    }
    element.setAttribute('style', styles);
  }

  return element;
}

function createSuperTable(data, attributes) {
  if (attributes === undefined) attributes = '';
  table = createSuperElement('table', attributes);

  for (var i in data) { // rows
    table.appendChild(createSuperElement('tr'));

    for (var j in data[i]) { // cells
      table.lastChild.appendChild(createSuperElement('td', data[i][j][1], data[i][j][0], data[i][j][2]));
    }
  }

  return table;
}

function resetRangeValues() {

  for (var i = 0; i < posesCharamel.length; i++) {
    //console.log(posesCharamel[i]);
    setSliderTo(posesCharamel[i], 0);

  }

}
// download the expression
var expressiondic = {};

function getExpression() {
  for (var i = 0; i < posesCharamel.length; i++) {
		var name = posesCharamel[i];
    try {
						var slider = document.getElementsByName(name)[0];
						expressiondic[name]= parseFloat(slider.value);
				} catch (error) {
					console.log(name+ ": is not there");}
		
		
  }
	console.log(JSON.stringify(expressiondic));
	return JSON.stringify(expressiondic);
  
}

function download(filename, text) {
  var element = document.createElement('a');
  element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
  element.setAttribute('download', filename);

  element.style.display = 'none';
  document.body.appendChild(element);

  element.click();

  document.body.removeChild(element);
}

// Start file download.
document.getElementById("dwn-btn").addEventListener("click", function() {
  // Generate download of hello.txt file with some content
  var text = getExpression();
  var filename = document.getElementById("text-val").value;

  download(filename, text);
}, false);


