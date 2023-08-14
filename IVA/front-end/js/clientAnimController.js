const WS_CONFIG = {
  protocol: "ws",
  hostname: "localhost",
  port: "8000",
  endpoint: "/ws/1",
};

const animationController = {
  webSocket: null,
  animationQueue: [],
  prevEulerAngles: { x: 0, y: 0, z: 0 },
  animCompleted: null,
  frameRate: 1000 / 32, // 1000 ms divided by 32 frames
  connect() {
    const { protocol, hostname, port, endpoint } = WS_CONFIG;
    const webSocketURL = `${protocol}://${hostname}:${port}${endpoint}`;
    this.webSocket = new WebSocket(webSocketURL);
    this.webSocket.onopen = this.onOpen.bind(this);
    this.webSocket.onclose = this.onClose.bind(this);
    this.webSocket.onerror = this.onError.bind(this);
    this.webSocket.onmessage = this.onMessage.bind(this);
  },
  disconnect() {
    if (this.webSocket) {
      this.webSocket.close();
    }
  },
  onOpen(event) {
    console.log("WebSocket OPEN:", event);
    this.handleAnimationLoop();
  },
  onClose(event) {
    console.log("WebSocket CLOSE:", event);
  },
  onError(event) {
    console.error("WebSocket ERROR:", event);
  },
  onMessage(event) {
    const wsMsg = event.data;
    if (wsMsg.includes("error")) {
      console.error("WebSocket Error:", wsMsg);
    } else {
      this.animationQueue.push(wsMsg);
    }
  },
  setExpressions(name, val) {
    vm.models.getFirst().setPoseByName(name, val);
  },
  setEuler(x, y, z) {
     vm.models.getFirst().createCtrlNode('head').setEuler(x, y, z);
  },
  animationHandler(wsMsg) {
    const jsonData = JSON.parse(wsMsg);
    const len = jsonData.length;
    let currentIndex = len;

    / Debugging: Print incoming JSON array information
    console.log("Incoming JSON array length:", jsonData.length);

    const animateSequence = () => {
      const i = len - currentIndex;
      currentIndex--;

      for (const [name, val] of Object.entries(jsonData[i].expressions)) {
        this.setExpressions(name, val);
      }

      const { globx: pitch, globy: yaw, globz: roll, jawx, jawy, jawz } = jsonData[i].poses;

      const deltaX = -yaw - this.prevEulerAngles.x;
      const deltaY = pitch - this.prevEulerAngles.y;
      const deltaZ = roll - this.prevEulerAngles.z;

      this.setEuler(deltaX, deltaY, deltaZ);

      this.prevEulerAngles = { x: deltaX, y: deltaY, z: deltaZ };

      if (currentIndex > 0) {
        setTimeout(animateSequence, this.frameRate);
      } else {
        this.animCompleted = true;
      }
    };

    animateSequence();
  },
  handleAnimationLoop() {
    if (this.animCompleted && this.animationQueue.length > 0) {
      this.animCompleted = false;
      const wsMsg = this.animationQueue.shift();
      this.animationHandler(wsMsg);
    }
    requestAnimationFrame(this.handleAnimationLoop.bind(this));
  },
  sendToWSServer(wsMessage) {
    if (this.webSocket && this.webSocket.readyState === WebSocket.OPEN) {
      this.webSocket.send(wsMessage);
    }
  },
};

//animationController.connect();

window.animationController = animationController;