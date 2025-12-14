gsap.from("nav",{y:-80,opacity:0,duration:1});
gsap.from(".hero h1",{x:-80,opacity:0,duration:1,delay:.3});
gsap.from(".hero p",{x:-60,opacity:0,duration:1,delay:.6});
gsap.from(".card",{y:40,opacity:0,stagger:.2,delay:.8});
gsap.from(".glass",{x:80,opacity:0,duration:1,delay:.5});

document.getElementById("mode").onclick=()=>{
    document.body.classList.toggle("neon");
};

const drop = document.getElementById("drop");
const fileInput = document.getElementById("file");
const preview = document.getElementById("preview");
const img = document.getElementById("img");
const skeleton = document.getElementById("skeleton");
const analyzeBtn = document.getElementById("analyzeBtn");
const changeBtn = document.getElementById("changeBtn");
const result = document.getElementById("result");

let selectedFile = null;

drop.onclick = () => fileInput.click();

fileInput.onchange = e => {
    const f = e.target.files[0];
    if(!f) return;
    
    selectedFile = f;

    drop.style.display = "none";
    preview.style.display = "block";
    
    img.style.display = "none";
    skeleton.style.display = "block";
    result.classList.remove("show");
    changeBtn.style.display = "block";

    setTimeout(() => {
        img.src = URL.createObjectURL(f);
        img.onload = () => {
            skeleton.style.display = "none";
            img.style.display = "block";
        }
    }, 800);
};

changeBtn.onclick = () => {
    fileInput.value = "";
    selectedFile = null;
    preview.style.display = "none";
    drop.style.display = "flex";
    changeBtn.style.display = "none";
    result.classList.remove("show");
};

analyzeBtn.onclick = async () => {
    if(!selectedFile) {
        alert("Please upload an image first!");
        return;
    }

    analyzeBtn.classList.add("processing");
    analyzeBtn.innerText = "Processing...";
    analyzeBtn.disabled = true;
    
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error("Server connection failed");

        const data = await response.json();

        document.getElementById("g").innerText = data.gender;
        document.getElementById("a").innerText = Math.round(data.age);
        document.getElementById("c").innerText = data.confidence + "%";
        
        result.classList.add("show");

    } catch (error) {
        console.error("Error:", error);
        alert("Failed to analyze image. Make sure server is running.");
    } finally {
        analyzeBtn.classList.remove("processing");
        analyzeBtn.innerText = "Analyze";
        analyzeBtn.disabled = false;
    }
};