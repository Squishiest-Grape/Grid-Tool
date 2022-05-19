const imgsrc = game.scenes.get(game.user.viewedScene).data.img
let ans = await autoGrid(imgsrc)

console.log(ans)

async function autoGrid(imgsrc) {
	let time = Date.now()

	// extract image data
	const texture = await loadTexture(imgsrc)
	const sprite = new PIXI.Sprite(texture)
	const img = canvas.app.renderer.extract.pixels(sprite)
	// 0-> 
	// ->1
	const [h,w] = [sprite.height,sprite.width]
	time = log('Loading Image',time)

	// edge detect
	const gray = simpleGray(img)
	const edge = simpleEdge(gray,h,w)
	time = log('Edge Detection',time)

	// hough transform
	const hough = simpleHough(edge,h,w,65,85,75)
	time = log('Hough Transform',time)

	// frequency content
	const res = 2
	const n = 2**(Math.ceil(Math.log2(hough[0].length*res)))
	const [FFTa,FFTr,FFTi] = fftn2(hough,n)
	let FFTavg = avgn2(FFTa)
	const r = Math.log(n,6)
	FFTavg = avg1(FFTavg,parseInt(r),r)
	residualCap(FFTa,FFTavg,0)
	time = log('Frequency Content',time)
	
	// angle detection
	let peaks = []
	let angles = []
	const std = 50*res
	for (let a=0; a<180; a++) {
		const peak = findPeaks(FFTa[a],std)
		peaks.push(peak)
		angles.push(peak.reduce((a,b)=>a+b,0))
	}
	time = log('Angle Detection',time)
	
	// detect angles
	
	
	
	return angles
	
	
}


function log(text,time=undefined) {
	if (time) {
		console.log(`Autogrid: ${text}: ${(Date.now() - time)/1000} sec`)
		time = Date.now()
		return time
	} else {
		console.log(`Autogrid: ${text}`)
	}
}

function simpleGray(img){
	const n = img.length
	const n4 = n*4
	let gray = new Float32Array(n)
	for (let ind=0; ind<n4; ind+=4){ gray[ind] = img[ind]*.299 + img[ind+1]*.587 + img[ind+2]*.114 }
	return gray
}

function simpleEdge(gray,h,w){
	const h1 = h - 1
	const w1 = w - 1
	let edge = new Float32Array(h1*w1)
	for (let j=0; j<h1; j++) {
		const wj = w*j
		const wj1 = wj + w
		for (let i=0; i<w1; i++) {
			const c0 = wj + i      //c 01
			const c1 = wj + i + 1  //c 23 
			const c2 = wj1 + i
			const c3 = wj1 + i + 1			
			const ev = gray[c2] + gray[c3] - gray[c0] - gray[c1]
			const eh = gray[c1] + gray[c3] - gray[c0] - gray[c2]
			edge[c1] = ev*ev + eh*eh
		}
	}
	return edge
}

function simpleHough(edge,h,w,minP,maxP,adjP){
	// simple constants
	const h1 = h - 1
	const w1 = w - 1
	const hr = h/2 - 1
	const wr = w/2 - 1
	const nr = Math.floor(((hr*hr+wr*wr)**.5)/2)+1
	const n = 2*nr+1
	// edge serach criteria
	let edgeSorted = edge.slice()
	edgeSorted.sort()
	const eMin = edgeSorted[Math.floor(minP*(edge.length-1)/100)]
	const eMax = edgeSorted[Math.ceil(maxP*(edge.length-1)/100)]
	// prepare angle arrays
	let S = new Float32Array(180)
	let C = new Float32Array(180)
	const d2r = Math.PI/180
	for (let a=0; a<180; a++) {
		S[a] = Math.sin(a*d2r)
		C[a] = Math.cos(a*d2r)
	}	
	// hough transform
	let hough = []
	for (let a=0; a<180; a++){ hough.push((new Float32Array(n)).fill(0)) }
	for (let j=0; j<h1; j++) {
		const wj = w*j
		for (let i=0; i<w1; i++) {
			const e = edge[wj+i]
			if ( (e < eMin) || (e > eMax) ) { continue }
			for (let a=0; a<180; a++) {
				const r = C[a]*(i-wr) + S[a]*(j-hr) + nr
				const i_r = Math.floor(r)
				const di = r - i_r
				hough[a][i_r] += 1 - di
				hough[a][i_r+1] += di
			}
		}
	}
	// normalize hough matrix
	const adjS = 1 - adjP/100
	const adjO = nr*(1 - adjP/200)	
	let p0 = NaN
	let p1 = NaN
	let p2 = NaN
	let p3 = NaN
	let dx = 0
	let dy = 0
	for (let a=0; a<180; a++) {
		const c = C[a]
		const s = S[a]
		for (let i=0; i<n; i++) {			
			const val = hough[a][i]
			if (val <= 0) { continue }			
			const r = i - nr
			p0 = NaN
			p1 = NaN
			p2 = NaN
			p3 = NaN
			if (s != 0) {
				p0 = (r + wr*c)/s
				p2 = (r - wr*c)/s
			} else {
				p0 = NaN
				p2 = NaN
			}
			if (c != 0) {
				p1 = (r - hr*c)/s
				p3 = (r + hr*c)/s
			} else {
				p1 = NaN
				p3 = NaN				
			}
			const g0 = (p0 >= -hr) && (p0 < hr)
            const g1 = (p1 >= -wr) && (p1 < wr)
            const g2 = (p2 > -hr) && (p2 <= hr)
            const g3 = (p3 > -wr) && (p3 <= wr)
			if 			(g0 && g1) {
				dx = -wr - p1
				dy = p0 - hr
			} else if 	(g0 && g2) {
				dx = -wr - wr
                dy = p0 - p2
			} else if 	(g0 && g3) {
				dx = -wr - p3
                dy = p0 + hr
			} else if 	(g1 && g2) {
				dx = p1 - wr
                dy = hr - p2
			} else if 	(g1 && g3) {
				dx = p1 - p3
                dy = hr + hr
			} else if 	(g2 && g3) {
				dx = wr - p3
                dy = p2 + hr
			} else {
				continue
			}
			const d = ((dx*dx + dy*dy)**.5)*adjS + adjO
			hough[a][i] = val/d
		}
	}
	return hough
}

// 2d fft
function fftn2(Xn,n) {
	const n2 = n/2
	const pi2 = -2*Math.PI/n
	const a_len = Xn.length
	let C = new Float32Array(n2)
	let S = new Float32Array(n2)
	for (let i=0; i<n2; i++) {
		C[i] = Math.cos(i*pi2)
		S[i] = Math.sin(i*pi2)
	}
	let FFTa = []
	let FFTr = []
	let FFTi = []
	for (let a=0; a<a_len; a++) {
		const [Fa,Fr,Fi] = fft(Xn[a],C,S,n)
		FFTa.push(Fa)
		FFTr.push(Fr)
		FFTi.push(Fi)
	}
	return [FFTa,FFTr,FFTi]
}

// 1d fft
function fft(X0,C,S,n) {
	const n2 = n/2
	let Fr = new Float32Array(n)
	let Fi = new Float32Array(n)
	let X = new Float32Array(n)
	X.fill(0)
	X.set(X0)
	fft2(Fr,Fi,C,S,X,0,0,1,n)
	Fr = Fr.slice(0,n2)
	Fi = Fi.slice(0,n2)
	let Fa = new Float32Array(n2)
	for (let i=0; i<n2; i++) { Fa[i] = (Fr[i]*Fr[i] + Fi[i]*Fi[i])**.5 }
	return [Fa,Fr,Fi]
}

// partial fft
function fft2(Fr,Fi,C,S,X,fs,xs,xi,n) {
    if (n === 2) {        
        const f2 = fs + 1
        const x2 = X[xs + xi]
        Fr[f2] = X[xs] - x2
        Fi[f2] = 0
        Fr[fs] = X[xs] + x2
        Fi[fs] = 0
	} else {
        const n2 = n/2
        const xi2 = xi*2
        fft2(Fr,Fi,C,S,X,fs,xs,xi2,n2)
        fft2(Fr,Fi,C,S,X,fs+n2,xs+xi,xi2,n2)
        const km = fs + n2
        for (let k=fs, a=0; k<km; k++, a+=xi) {
            const k2 = k + n2
            const qr = C[a]*Fr[k2] - S[a]*Fi[k2]
            const qi = C[a]*Fi[k2] + S[a]*Fr[k2]
            Fr[k2] = Fr[k] - qr
            Fi[k2] = Fi[k] - qi
            Fr[k] += qr
            Fi[k] += qi
		}
	}
}

// 2d average along 1st axis
function avgn2(X) {
	const m = X.length
	const n = X[0].length
	let A = new Float32Array(n)
	for (let i=0; i<n; i++) {
		A[i] = 0
		for (let j=0; j<m; j++) { A[i] += X[j][i] }
		A[i] /= m
	}
	return A
}

// 1d average with multiplier
function avg1(X,r,m) {
	const n = X.length
	const d = (2*r+1)/m
	const r1 = r+1
	let A = new Float32Array(n)
	for (let i=0; i<r; i++) {
		A[i] = 0
		for (let j=0; j<i+r1; j++) { A[i] += X[j] }
		A[i] += (r-i)*X[0]
		A[i] /= d
	}
	for (let i=r; i<n-r; i++) {
		A[i] = 0
		for (let j=i-r; j<i+r1; j++) { A[i] += X[j] }
		A[i] /= d
	}
	for (let i=n-r; i<n; i++) {
		A[i] = 0
		for (let j=i-r; j<n; j++) { A[i] += X[j] }
		A[i] += (r-i)*X[n-1]
		A[i] /= d
	}
	return A
}

// calculate residual
function residualCap(X,A,min) {
	const m = X.length
	const n = A.length
	for (let j=0; j<m; j++) {
		for (let i=0; i<n; i++) {
			X[j][i] -= A[i]
			if (X[j][i] < min) { X[j][i] = min }	
		}
	}
}


// peak detection
function findPeaks(vals,std){
	const n = vals.length
	const n2 = n/2
	const d = 2*(std)**2
	let peaks = new Uint8Array(n)
	peaks.fill(1)
	let eq = new Float32Array(n2)
	for (let x=0; x<n2; x++) { eq[x] = Math.E**((-(x**2)/d)) }
	for (let i=0; i<n; i++){
		if (peaks[i]) {
			if (vals[i] > 0) {
				for (let x=1; x<n2; x++) {
					const y = vals[i]*eq[x]
					let j = (i-x)%n
					if (y > vals[j]) { peaks[j] = 0 }
					j = (i+x)%n
					if (y > vals[j]) { peaks[j] = 0 }
				}
			}
		}
	}
	let ans = []
	for (let i=0; i<n; i++) { if (peaks[i]) { ans.push(i) }  }
	return ans
}