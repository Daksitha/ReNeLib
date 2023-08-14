
/**
 * Global variables
 */
var webSocket   = null;
var ws_protocol = "ws";
var ws_hostname = "localhost";
var ws_port = "8000";
var ws_endpoint ="/ws/1";

var aliveTimer = null;

let prev_x = 0.0;
let prev_y = 0.0;
let prev_z = 0.0;


var fps = 18;
var speed = 1000/fps;
// add incoming data into the queue
var queue = [];
var anim_completed = null;
/**
 * Event handler for clicking on button "Connect"
 */
function connect() {
    openWSConnection(ws_protocol, ws_hostname, ws_port, ws_endpoint);

}
/**
 * Event handler for clicking on button "Disconnect"
 */
function disconnect() {
    webSocket.close();
}

/**
 * Message from client (this and webpage loading this) that client is alive
 */
function clientAliveMessage() {
    //console.log("Send client alive message to server");
    sendToWSServer("gui_alive");
    aliveTimer = setTimeout(clientAliveMessage, 2000);
}
/**
 * Message from client to send animations
 */
function clientRequestAnimation() {
   // console.log("request_server_animations");
    sendToWSServer("send_animations");

}

/**
 * measure time elapsed
 */
var startTime, endTime;

function start() {
    startTime = new Date();


}

function end() {
    endTime = new Date();
    var timeDiff = endTime - startTime; //in ms
    // strip the ms
    timeDiff /= 1000;

    // get seconds
    var seconds = timeDiff;
    //console.log(seconds + " seconds");
    //document.getElementById("incomingMsgOutput").value += "time_elapsed: " + seconds.toString() +"seconds"+ "\r\n";

}

/**
 * set expression and
 */
function setExpressions(name,val){
    vm.models.getFirst().setPoseByName(name, val);
}
// function setHeadPose(neckpitch,neckyaw){
//     let curr_pitch = neckpitch;
//     let curr_yaw = neckyaw;
//
//     let glob_pitch = curr_pitch-prev_pitch;
//     let glob_yaw = curr_yaw- prev_x;
//
//     vm.models.getFirst().createCtrlNode('head').setEuler(glob_yaw, glob_pitch, 0.0);
//     prev_pitch = curr_pitch;
//     prev_x = curr_yaw;
// }
// function toRadians (angle) {
//     return angle * (Math.PI / 180);
// }

/**
 * this function send the charamel character a sequence of animations
 *
 *             FLAME: (neck bone with global)
 *             x: pitch. positive for looking down.
 *             y: yaw. positive for looking left.
 *             z: roll. positive for tilting head right.
 *             Charamel: (head bone or neck bone ?)
 *             setEuler
 *             x: yaw.  positive for looking right = -flame_yaw
 *             y: pitch. positive for look down = flame_pitch
 *             z: roll. positive for tilting head right = flame_roll
 *
 *
 */

function animationHandler(wsMsg) {
  try {
    const jasonData = JSON.parse(wsMsg);
    const len = Object.keys(jasonData).length;
    let indexing = len;

    const rendering_anim_sequence_fps = () => {
      const startTime = performance.now();
      let i = len - indexing;
      indexing--;
      for (const [name, val] of Object.entries(jasonData[i].expressions)) {
        setExpressions(name, val);
      }

      const { globx: flame_pitch, globy: flame_yaw, globz: flame_roll, jawx:jawopen, jawy, jawz } = jasonData[i].poses;
      // x: yaw = -flame_yaw// y: pitch = flame_pitch  // z: roll = flame_roll
        let rounded_flame_pitch = parseFloat(flame_pitch.toFixed(2));
        let rounded_flame_yaw = parseFloat(flame_yaw.toFixed(2));
        let rounded_flame_roll = parseFloat(flame_roll.toFixed(2));

        // x: yaw = -flame_yaw// y: pitch = flame_pitch  // z: roll = flame_roll
        let char_x = -rounded_flame_yaw - prev_x;
        let char_y = rounded_flame_pitch - prev_y;
        let char_z = rounded_flame_roll - prev_z;

        // jawy: positive --> flame_jawright, negative --> flamejawleft
        vm.models.getFirst().createCtrlNode('head').setEuler(char_x, char_y, char_z);
        vm.models.getFirst().setPoseByName('jawopen', Math.min(Math.max(jawopen, 0), 1));
         if (jawz >= 0) {
          vm.models.getFirst().setPoseByName('jawleft', Math.min(Math.max(jawz, 0), 1));
        } else {
          vm.models.getFirst().setPoseByName('jawright', Math.min(Math.max(jawz, 0), 1));
        }


        prev_z = rounded_flame_roll;
        prev_x = -rounded_flame_yaw ;
        prev_y = rounded_flame_pitch;


      /* let char_x = -flame_yaw - prev_x;
      let char_y = flame_pitch - prev_y;
      let char_z = flame_roll - prev_z;
      vm.models.getFirst().createCtrlNode('head').setEuler(char_x, char_y, char_z);
      prev_z = flame_roll;
      prev_x = -flame_yaw ;
      prev_y = flame_pitch;*/



      if (indexing > 1) {
        setTimeout(rendering_anim_sequence_fps, speed);
      } else {
        anim_completed = true;
      }
      if (indexing === 10) {
        sendToWSServer("send_animations");
      }
      const endTime = performance.now();
      console.log(`One anim seq play ${endTime - startTime} milliseconds`);
    };
    rendering_anim_sequence_fps();
  } catch (exception) {
    console.error(exception);
  }
}

/**
 * loop through the queue and play if there are animations
 * incoming messages are queued and this act as a loop to play
 * animations one after the other
 */
function clientRequestAnimationSetTime() {
    //document.getElementById("incomingMsgOutput").value += "clientRequestAnimationSetTime: " + "\r\n";
    if(anim_completed){
        //document.getElementById("incomingMsgOutput").value += "Fetching animation from the queue leng: " + queue.length.toString() + "\r\n";
        if (queue.length !== 0){
            anim_completed = false;
            // get the first element
            let q_ws_Msg = queue[0];
            animationHandler(q_ws_Msg);
            if(queue.length >= 1) {
                //document.getElementById("incomingMsgOutput").value += "removing played anim from queue, now leng: " + queue.length.toString() + "\r\n";
                queue.shift();
            }
            //document.getElementById("incomingMsgOutput").value += "played the animation, now queue leng: " + anim_completed.toString() + "\r\n";
        }
        else{
            //document.getElementById("incomingMsgOutput").value += "Oh, the queue is empty, try again, queue leng: " + queue.length.toString() + "\r\n";
        }



    }
    //sendToWSServer("send_animations");
    aliveTimer = setTimeout(clientRequestAnimationSetTime, 1);

}
/**
 * Open a new WebSocket connection using the given parameters
 */
function openWSConnection(protocol, hostname, port, endpoint) {
    var webSocketURL;
    webSocketURL = protocol + "://" + hostname + ":" + port + endpoint;
    console.log("openWSConnection to: " + webSocketURL);
    try {
        webSocket = new WebSocket(webSocketURL);
        webSocket.onopen = function(openEvent) {
            console.log("WebSocket OPEN: " + JSON.stringify(openEvent, null, 4));
            //document.getElementById("incomingMsgOutput").value += "WebSocket OPEN: " + JSON.stringify(openEvent, null, 4);
            //clientAliveMessage();
            clientRequestAnimationSetTime();


        };
        webSocket.onclose = function (closeEvent) {
            console.log("WebSocket CLOSE: " + JSON.stringify(closeEvent, null, 4));
            clearTimeout(aliveTimer);
        };
        webSocket.onerror = function (errorEvent) {
            console.log("WebSocket ERROR: " + JSON.stringify(errorEvent, null, 4));
        };
        webSocket.onmessage = function (messageEvent) {
            let wsMsg = messageEvent.data;
            console.log("WebSocket MESSAGE: " + wsMsg.length);

            if (wsMsg.indexOf("error") > 0) {
                //document.getElementById("incomingMsgOutput").value += "error: " + wsMsg.error + "\r\n";
            } else {
                //document.getElementById("incomingMsgOutput").value += "A message arrived: " + "\r\n";
                if (anim_completed == null){
                    anim_completed = false;
                    animationHandler(wsMsg);
                    //document.getElementById("incomingMsgOutput").value += "First animation: " + anim_completed.toString() + "\r\n";
                }
                else {
                    queue.push(wsMsg)
                    //document.getElementById("incomingMsgOutput").value += "Animation still in play, Queueing: " + queue.length.toString() + "\r\n";

                }




            }
        };


    } catch (exception) {
        console.error(exception);
    }
}

/**
 * Send a message to the WebSocket server
 */
function sendToWSServer(ws_message) {
    if (typeof webSocket == 'undefined') {
        return;
    }
    if (webSocket.readyState != WebSocket.OPEN) {
        console.error("webSocket is not open: " + webSocket.readyState);
        return;
    }
    webSocket.send(ws_message);
}
