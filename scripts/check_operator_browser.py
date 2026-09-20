"""Optional smoke test against a fresh local server; no real gamepad emulation.

pip install -r requirements-operator-test.txt
python -m playwright install chromium
python scripts/check_operator_browser.py
Restart the server afterwards to release this test browser's operator lease.
"""
import asyncio
import os
from playwright.async_api import async_playwright


async def main():
    async with async_playwright() as p:
        options = dict(headless=True, args=['--no-sandbox', '--use-gl=angle',
                                           '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        if os.getenv('CHROMIUM_EXECUTABLE'):
            options['executable_path'] = os.environ['CHROMIUM_EXECUTABLE']
        browser = await p.chromium.launch(**options)
        page = await browser.new_page(viewport=dict(width=1440, height=1100))
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        await page.goto(os.getenv('OPERATOR_URL', 'http://127.0.0.1:8000'))
        await page.wait_for_function("document.querySelector('#connection').textContent.includes('接続済み')")
        await page.click('#create')
        await page.wait_for_function("document.querySelector('#lifecycle').textContent==='READY'")
        await page.click('#start')
        await page.wait_for_function("document.querySelector('#lifecycle').textContent==='RUNNING'")
        await page.wait_for_function("Number(document.querySelector('#altitude').textContent)>24", timeout=30000)
        await page.wait_for_timeout(3000)
        # Match the live automatic commands using the keyboard-mode sliders.
        for _ in range(80):
            await page.evaluate("""() => {
                for(const name of ['throttle','pitch_deg','bank_deg','rudder']) {
                    const value=Number(document.querySelector('#value-'+name).textContent.split('/')[1]);
                    const element=document.querySelector('#axis-'+name);
                    element.value=value; element.dispatchEvent(new Event('input',{bubbles:true}));
                }
            }""")
            await page.wait_for_timeout(100)
            if '0.5/0.5' in await page.locator('#instruction').inner_text():
                break
        assert '0.5/0.5' in await page.locator('#instruction').inner_text(), 'handover did not become available'
        await page.click('#take')
        await page.wait_for_function("document.querySelector('#authority').textContent.includes('YOU HAVE CONTROL')")
        await page.click('#pause')
        await page.wait_for_function("document.querySelector('#lifecycle').textContent==='PAUSED'")
        clock = await page.locator('#clock').inner_text()
        await page.wait_for_timeout(700)
        assert clock == await page.locator('#clock').inner_text()
        await page.click('#resume')
        await page.wait_for_function("document.querySelector('#lifecycle').textContent==='RUNNING'")
        await page.click('#abort')
        await page.wait_for_function("document.querySelector('#lifecycle').textContent==='ABORTED'")
        assert await page.locator('#review').is_visible()
        await page.set_viewport_size(dict(width=390, height=844))
        assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        assert not errors, errors
        print('PASS: browser create / takeoff / manual handover / pause / resume / abort / responsive layout')
        await browser.close()


if __name__ == '__main__':
    asyncio.run(main())
