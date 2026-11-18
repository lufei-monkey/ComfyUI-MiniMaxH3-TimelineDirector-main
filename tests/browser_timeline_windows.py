"""Run against an isolated ComfyUI test server, never a user's open workflow."""
import json
import sys
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright

url = sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:8893'
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,channel='msedge')
    page=browser.new_page(viewport={'width':1800,'height':1200})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(url)
    page.wait_for_function("()=>!!window.comfyAPI?.app?.app?.graph && !!window.LiteGraph?.registered_node_types?.MiniMaxH3TimelinePlanner",timeout=90000)
    page.evaluate("""async()=>{
      const {app}=await import('/scripts/app.js');window.testApp=app;
      await app.ui.settings.setSettingValue('Comfy.VueNodes.Enabled',true);
      app.graph.clear();const n=LiteGraph.createNode('MiniMaxH3TimelinePlanner');app.graph.add(n);n.pos=[50,50];
      window.testNode=n;app.canvas.ds.scale=0.9;app.canvas.ds.offset=[0,0];app.canvas.setDirty(true,true);
    }""")
    page.wait_for_function('()=>window.testNode?.__m3td?.stage?.isConnected')
    page.locator('[data-field="segmentCount"]').fill('4')
    page.locator('[data-action="applySegments"]').click()
    page.wait_for_timeout(500)
    assert page.locator('[data-window]').count()==4
    assert page.locator('[data-segment-prompt]').count()==1
    page.locator('[data-segment-prompt]').fill('Segment one: a quiet room.')
    page.locator('[data-select-segment="1"]').click()
    page.locator('[data-segment-prompt]').fill('Segment two: a person enters.')
    before=page.evaluate('testNode.__m3td.state.segmentConfig.segments')
    handle=page.locator('[data-window="1"] [data-edge="right"]')
    box=handle.bounding_box();page.mouse.move(box['x']+box['width']/2,box['y']+box['height']/2)
    page.mouse.down();page.mouse.move(box['x']+box['width']/2+65,box['y']+box['height']/2,steps=15);page.mouse.up()
    after=page.evaluate('testNode.__m3td.state.segmentConfig.segments')
    assert after[1]['endFrame']!=before[1]['endFrame'],(before,after)
    assert after[2]['startFrame']-before[2]['startFrame']==after[1]['endFrame']-before[1]['endFrame']
    for edge,delta in [('move',50),('left',-50)]:
        h=page.locator(f'[data-window="1"] [data-edge="{edge}"]').bounding_box()
        old=page.evaluate('testNode.__m3td.state.segmentConfig.segments')
        page.mouse.move(h['x']+h['width']/2,h['y']+h['height']/2);page.mouse.down()
        page.mouse.move(h['x']+h['width']/2+delta,h['y']+h['height']/2,steps=12);page.mouse.up()
        new=page.evaluate('testNode.__m3td.state.segmentConfig.segments')
        assert new[1]['startFrame']!=old[1]['startFrame'],(edge,old,new)
    # Exact touching via the numeric start control, preserving every output frame.
    page.locator('[data-field="selectionStart"]').fill(str(after[0]['endFrame']/24))
    page.locator('[data-field="selectionStart"]').press('Tab')
    touching=page.evaluate('testNode.__m3td.state.segmentConfig.segments')
    assert touching[1]['startFrame']==touching[0]['endFrame']
    page.locator('[data-select-segment="0"]').click()
    assert page.locator('[data-segment-prompt]').input_value()=='Segment one: a quiet room.'
    page.evaluate("""()=>{const u=testNode.__m3td;
      u.state.images=[{id:'test-image',file:'test-image.png',name:'Test image'}];
      u.state.audios=[{id:'test-audio',file:'test-audio.wav',name:'Test audio'}];
      u.render();}
    """)
    for kind,asset in [('images','test-image'),('audios','test-audio')]:
        page.locator(f'[data-asset-id="{asset}"]').drag_to(page.locator(f'[data-segment-drop="{kind}"]'))
        assert page.locator(f'[data-segment-asset="{asset}"]').count()==1
    page.screenshot(path=str(Path(tempfile.gettempdir())/'h3_segment_windows_ui.png'),full_page=True)
    # Reconfigure restores windows and prompts from the serialized workflow.
    page.evaluate('()=>{const v=testApp.graph.serialize();testApp.graph.configure(v);window.testNode=testApp.graph._nodes[0]}')
    page.wait_for_timeout(500)
    assert page.evaluate('testNode.__m3td.state.segmentConfig.segments[1].prompt')=='Segment two: a person enters.'
    page.locator('[data-field="segmentCount"]').fill('0');page.locator('[data-action="applySegments"]').click()
    page.wait_for_timeout(500)
    assert page.locator('[data-window]').count()==0
    assert page.locator('[data-segment-prompt]').count()==0
    heights=page.evaluate('()=>{const u=testNode.__m3td;return {root:u.root.offsetHeight,foot:u.requiredDirectorHeight()}}')
    assert abs(heights['root']-heights['foot'])<12,heights
    print(json.dumps({'passed':True,'heights':heights,'pageErrors':errors},ensure_ascii=False))
    browser.close()
