const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const root=path.join(__dirname,'../visualizations/author_network'),elements=new Map();
function fake(){return {children:[],style:{},classList:{contains:()=>false},textContent:'',value:'',
set innerHTML(v){this.children=[];},get innerHTML(){return '';},setAttribute(){},addEventListener(){},
append(...els){this.children.push(...els);},appendChild(el){this.children.push(el);},replaceChildren(...els){this.children=els;}};}
function element(id){if(!elements.has(id)){const e=fake();e.value=id==='sentiment-select'?'net':id==='emotion-select'?'joy':'1';elements.set(id,e);}return elements.get(id);}
const context=vm.createContext({document:{addEventListener(){},getElementById:element,createElement:fake,querySelectorAll:()=>[]},window:{},Float32Array,Set,Map,Number,console});
const run=s=>vm.runInContext(s,context);
run(fs.readFileSync(path.join(root,'app.js'),'utf8'));
const data=Array.from({length:6},(_,i)=>({id:String(i+1),dt:[0,0,0,1,1,2][i],c:[2,2,0,0,1,9][i],
t0:i<3?.8:.1,t1:i===3||i===4?.8:.1,t2:i===5?.8:.1,sp:.3,sn:.4,sg:.3,ea:.2,eb:.2,ec:.2,ed:.2,ee:.2,ef:.2,eg:.2,eh:.2,ei:.2,ej:.2,ek:.2}));
const before=JSON.stringify(data);context.dataset=data;
run("nodes=dataset;communityLabels={'0':{label:'Zero'},'1':{label:'One'},'2':{label:'Two'}};TOPIC_NAMES=['A','B','C'];TOPIC_PALETTE=[[1,0,0],[0,1,0],[0,0,1]];visible3D=new Float32Array([1,1,1,1,0,0]);colors=new Float32Array(nodes.length*3);initialiseFocusControls();");
assert.deepEqual(Array.from(run('topicOrder')),[0,1,2]);
function check(predicate){assert.deepEqual(Array.from(run('focusMask')),data.map(n=>predicate(n)?1:0));}
run("toggleGroup('topic',0,true);toggleGroup('topic',2,true)");check(n=>[0,2].includes(n.dt));
assert.equal(run('focusTotals.count'),4);
for(const mode of ['community','dominant_topic','sentiment','topic_membership','emotion']){
run("colorMode='"+mode+"';updateColors()");check(n=>[0,2].includes(n.dt));
assert.ok(Array.from(run('colors')).every(Number.isFinite));
if(['community','dominant_topic'].includes(mode)){
const rows=element('legend').children,counts=rows.map(r=>Number(r.children[3].textContent.replaceAll(',','')));
assert.deepEqual(counts,[...counts].sort((a,b)=>b-a));assert.equal(counts.reduce((a,b)=>a+b,0),6);
assert.equal(rows.filter(r=>r.children[0].checked).length,mode==='dominant_topic'?2:0);
}}
run("toggleGroup('community','0',true)");check(n=>n.c===0);
run("toggleGroup('community','2',true)");check(n=>[0,2].includes(n.c));
run("toggleGroup('topic',1,true)");check(n=>n.dt===1);
run("toggleGroup('topic',1,false)");check(()=>true);
run("toggleGroup('community','tail',true)");check(n=>n.c===9);
run("clearFocus()");check(()=>true);assert.equal(JSON.stringify(data),before);
run("examplePostsState='ready';examplePosts={'1':['123','retweet_max_distinct_tokens','retweeted','<script>bad</script>']}");
assert.ok(run("examplePostHtml({id:'1'})").includes('&lt;script&gt;'));
assert.ok(run("examplePostHtml({id:'1'})").includes('Retweet — shared content'));
const calls=[];context.capture=p=>calls.push(p.focusPass);context.depth=()=>calls.push('depth');
run("toggleGroup('topic',0,true);regl={clear:depth};drawFocusedCloud(capture,{},true)");
assert.deepEqual(calls,[0,'depth',1]);
const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
for(const removed of ['More options','Focus on authors','id="legend-tabs"'])assert.ok(!html.includes(removed));
const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);assert.equal(ids.length,new Set(ids).size);
console.log('PASS: portable 3-topic/6-author viewer; sorted legends, retained selection, clear, safe excerpts, 3D focus order.');
