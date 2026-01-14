let paused = false;
let video = document.getElementById("video");

function updateVideo() {
  if (!paused) {
    video.src = "/video_frame?" + new Date().getTime();
  }
}
setInterval(updateVideo, 100); // 10 FPS, NO flicker

function capture() {
  paused = true;

  fetch(video.src)
    .then(res => res.blob())
    .then(blob => {
      let formData = new FormData();
      formData.append("frame", blob);

      fetch("/inspect", {
        method: "POST",
        body: formData
      })
      .then(res => {
        const status = res.headers.get("X-Result");
        return res.blob().then(imgBlob => ({ status, imgBlob }));
      })
      .then(data => {
        // Show overlay image
        document.getElementById("result").src =
          URL.createObjectURL(data.imgBlob);

        // Show PASS / FAIL
        document.getElementById("status").innerText =
          "RESULT: " + data.status;
      });
    });
}


function resume() {
  paused = false;
  document.getElementById("status").innerText = "";
}
