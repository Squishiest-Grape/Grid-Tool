/*-------------------------------------------------------------------\\
                             Data
\\-------------------------------------------------------------------*/

// constants
const s = game.scenes.get(game.user.viewedScene)	// current scenes
const g_old = s.data.size                       	// original grid size
const W_old = s.data.width                      	// original width
const H_old = s.data.height                     	// original height
let w,h                                         	// image width,height
try { [w,h] = await img_dims(s.data.img) }
catch(error) { console.log(error); return }

// dynamic
let t = 0.01                                         // ratio search tollerance
let g = 0                                            // grid size choice (0=scale based on image)
let [wr,hr] = get_dims(w,h,t)                        // width, height, ratio
let k = 1                                            // grid scaleing value			
if (Math.min(wr,hr) < 10){ let k = Math.max(Math.round(W_old/g_old/wr),1) }


let W,H,g_											// actual dimensions to push
[W,H,g_,g] = get_dims2(w,h,wr,hr,g,k)

let save = false


/*-------------------------------------------------------------------\\
                              HTML
\\-------------------------------------------------------------------*/

const title = `Auto Scene Adjustment`

const content = () => // let this be a table because I don't know formating
`<table id="${title}">
	<tr>
		<th width="30%"><label>Detected Ratio:</label></th>
		<th><label updateChange=1>${wr}:${hr}</label></th>
		<th><label>Tollerance:</label></th>
		<th width="10%"><input updateChange=5 class="changer" type="text" id="tText" value="${t*100}%"></th>
	</tr>
    <tr>
		<th><label>Scaling Factor:</label></th>
		<th>
			<table><tr>
			<th><input updateChange=2 class="changer" type="range" id="kRange" value="${k}" min="${1}" max="${k*2+1}"></th>
			<th width="10%"><input updateChange=3 class="changer" type="text" id="kText" value="${k}"></th>
			</tr></table>
		</th>
		<th><label>Grid Size:</label></th>
		<th><input updateChange=4 class="changer" type="text" id="gText" value="${g?g:''}"></th>
    </tr>
</table>`


const buttons = {
	yes: {
		icon: "<i class='fas fa-check'></i>",
        label: `Set`,
        callback: ()=>{save=true}
	},
	no: {
		icon: "<i class='fas fa-times'></i>",
		label: `Cancel`,
		callback: ()=>{},
	},
}

/*-------------------------------------------------------------------\\
                           Functions
\\-------------------------------------------------------------------*/

// float capable gcd solver (euclid's algorithm is insufficent)
function fgcd(a,b,tol=0) {
	if (b>a) { [a,b] = [b,a] }
	for (let i=1; i<=b; i++) {
		let j = (i*a)/b
		if (Math.abs(j-Math.round(j)) < tol) { return b/i }
	}
	return 1
}

// get image dimensions from url
function img_dims(url) {
	return new Promise((resolve,reject) => {
		const img = new Image()
		img.onload = () => {
			const { naturalWidth: width, naturalHeight: height } = img
	        resolve([width, height])
		}
		img.onerror = () => {
			reject(`Error: Unable to load ${url} as image`)
		}
		img.src = url
	})
}

// get new image dimensions
function get_dims(w,h,t) {
	let f = fgcd(w,h,t)
	let wr = Math.round(w/f)
	let hr = Math.round(h/f)
	return [wr,hr]
}

function get_dims2(w,h,wr,hr,g,k){
	let g_ = g
	if (g==0){
		g_ = Math.round(w/wr/k) 
		if (g_ < 50) {
			g = 50
			g_ = 50 
		}
	}

	if (g!=0){
		var W = Math.round(wr*g_*k)
		var H = Math.round(hr*g_*k)
	} else {
		var W = w
		var H = h
	}
	return [W,H,g_,g]
}

function change(e){
	switch(e.target.id){
		case 'tText':
			t = e.target.value;
			if (t.slice(-1)=='%') { t = t.slice(0,-1); }
			t = Roll.safeEval(t)/100;
			[wr,hr] = get_dims(w,h,t);
			break;
		case 'kRange':
		case 'kText':
			k = Roll.safeEval(e.target.value);
			break;
		case 'gText':
			g = Roll.safeEval(e.target.value);
			if (!g) { g = 0; }
			break;
	}
	
	
	[W,H,g_,g] = get_dims2(w,h,wr,hr,g,k)
	
	g = Math.round(g)
	
	document.querySelector("[updateChange='1']").innerText = `${wr}:${hr}`
	document.querySelector("[updateChange='2']").value = k
	document.querySelector("[updateChange='2']").max = k*2+1
	document.querySelector("[updateChange='3']").value = k
	document.querySelector("[updateChange='4']").value = g?g:''
	document.querySelector("[updateChange='5']").value = `${t*100}%`
	
	s.update({height:H,width:W,grid:g_})
	console.log(`Scene dimesions updated to ${W}:${H}:${g_}`)
}

function close(){
	if (save) {
		console.log(`Scene dimesions saved to ${W}:${H}:${g_}`)
	} else {
		s.update({height:H_old,width:W_old,grid:g_old})
		console.log(`Scene demensions remains at ${W_old}:${H_old}:${g_old}`)
	}
}

// dialog creation function
function dialog() {
	let d = new Dialog({
		title:   title,     // add title
		content: content(), // add main body
		buttons: buttons,   // add terminating callback buttons
		close: close,
	},{jQuery:true}).render(true)
	// attach non-terminating event handlers
	$(document).ready(function() {
		for (let element of document.getElementsByClassName('changer')) {
			element.addEventListener('change',change)
		}
		// console.log(document.getElementById(title).childNodes)
	})
	return d
}

// /*-------------------------------------------------------------------\\
//                             Main Code
// \\-------------------------------------------------------------------*/

// // initilize screen update
s.update({height:H,width:W,grid:g_})

// // start dialog
let d = dialog()