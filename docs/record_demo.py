# GIF для README і відео-демо: Playwright відкриває запущений `findex serve` і робить
# те, що зробила б людина. Запуск:
#   uv run findex serve --index index.bin --port 8000 &
#   uv run --with playwright python docs/record_demo.py http://127.0.0.1:8000
#   (потім ffmpeg робить mp4 з webm, команда — в кінці скрипта)

import asyncio
import io
import sys
from pathlib import Path

from PIL import Image
from playwright.async_api import Page, async_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
OUT = Path(__file__).parent
SIZE = {"width": 1100, "height": 720}

CAPTION_JS = """
(text) => {
  let el = document.getElementById('demo-caption');
  if (!el) {
    el = document.createElement('div');
    el.id = 'demo-caption';
    el.style.cssText = 'position:fixed;left:50%;bottom:24px;transform:translateX(-50%);' +
      'max-width:900px;padding:12px 20px;border-radius:10px;background:rgba(20,20,30,.88);' +
      'color:#fff;font:500 18px/1.4 system-ui,sans-serif;z-index:99999;text-align:center';
    document.body.appendChild(el);
  }
  el.textContent = text;
  el.style.display = text ? 'block' : 'none';
}
"""


async def caption(page: Page, text: str, wait: float = 0) -> None:
    await page.evaluate(CAPTION_JS, text)
    if wait:
        await page.wait_for_timeout(wait * 1350)


async def search(page: Page, query: str, mode: str | None = None, delay: int = 70) -> None:
    box = page.locator("input[name=q]")
    await box.fill("")
    await box.type(query, delay=delay)
    if mode is not None and await page.locator("select[name=mode]").count():
        await page.select_option("select[name=mode]", mode)
    await page.click("button[type=submit]")
    await page.wait_for_selector(".hit, .notice")


# ---------------------------------------------------------------- GIF


async def gif(browser) -> None:  # type: ignore[no-untyped-def]
    page = await browser.new_page(viewport=SIZE, device_scale_factor=1)
    await page.goto(BASE)
    frames: list[Image.Image] = []
    durations: list[int] = []

    async def snap(ms: int) -> None:
        png = await page.screenshot()
        frames.append(Image.open(io.BytesIO(png)).convert("RGB").resize((880, 576)))
        durations.append(ms)

    await snap(900)
    box = page.locator("input[name=q]")
    for i, ch in enumerate("whale OR ship"):
        await box.type(ch)
        if i % 2:
            await snap(90)
    await page.click("button[type=submit]")
    await page.wait_for_selector(".hit")
    await page.wait_for_timeout(300)
    await snap(2200)
    await page.select_option("select[name=mode]", "semantic")
    await box.fill("")
    for i, ch in enumerate("obsessive skipper's vendetta"):
        await box.type(ch)
        if i % 3 == 2:
            await snap(80)
    await page.click("button[type=submit]")
    await page.wait_for_selector(".hit")
    await page.wait_for_timeout(300)
    await snap(3500)
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(
        OUT / "demo.gif",
        save_all=True,
        append_images=pal[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    await page.close()


# ---------------------------------------------------------------- відео


async def video(browser) -> None:  # type: ignore[no-untyped-def]
    ctx = await browser.new_context(viewport=SIZE, record_video_dir=str(OUT / "video"),
                                    record_video_size=SIZE)
    page = await ctx.new_page()
    await page.goto(BASE)
    await caption(page, "findex — пошуковик по 25 класичних книжках, написаний з нуля на Python", 6)
    await caption(page, "Мова запитів: AND (можна не писати), OR, NOT, дужки і фрази в лапках", 5)

    await search(page, "whale OR ship", "lexical")
    await caption(page, "За словами: Boolean-фільтр → BM25 на масивах NumPy → топ-10 через argpartition", 7)
    await caption(page, "Сніпет — вікно з найбільшою кількістю різних слів запиту, слова підсвічено", 6)
    await page.mouse.wheel(0, 500)
    await page.wait_for_timeout(2500)
    await page.mouse.wheel(0, -500)

    await search(page, '"white whale"', "lexical")
    await caption(page, "Фраза в лапках: позиції слів зберігаються в індексі (масив int32)", 5)
    await page.click(".hit a.title")
    await page.wait_for_load_state()
    await caption(page, "Перегляд книжки відкривається на тому самому входженні, що й у сніпеті", 6)
    await page.go_back()
    await page.wait_for_selector(".hit")

    await search(page, "obsessive skipper's vendetta", "lexical")
    await caption(page, "Цих слів у «Мобі Діку» немає взагалі — пошук за словами нічого не знаходить", 6)
    await search(page, "obsessive skipper's vendetta", "semantic", delay=20)
    await caption(page, "За змістом: 20 тис. фрагментів по 150 слів, кожен — вектор all-MiniLM-L6-v2", 7)
    await caption(page, "Косинус до всіх фрагментів — одне множення матриці на вектор; книжці — її найкращий фрагмент", 8)

    await search(page, "scientist reanimates a cadaver", "semantic", delay=40)
    await caption(page, "Ще перефразований запит: жодного слова з «Франкенштейна»", 6)

    await search(page, "ghost OR spirit", "hybrid", delay=40)
    await caption(page, "Гібрид: рейтинг BM25 і семантичний зливаються через Reciprocal Rank Fusion", 8)

    await page.goto(f"{BASE}/search?q=scientist+reanimates+a+cadaver&mode=semantic&k=3")
    await caption(page, "Те саме через HTTP API (FastAPI + Pydantic): /search?q=...&mode=semantic", 7)
    await caption(page, "У JSON — сніпет, межі підсвічених слів, бал і час; Swagger UI — на /api/docs", 6)
    await page.goto(f"{BASE}/stats")
    await caption(page, "/stats: 25 документів, 54 тис. термів, 3 млн токенів", 5)
    await page.goto(BASE)
    await caption(page, "Код, виміри лаб 1–8 і обмеження — у README репозиторію danilk202/findex", 6)
    await ctx.close()


async def main() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        await gif(browser)
        if "--video" in sys.argv:
            await video(browser)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
    # mp4 з webm: ffmpeg -i docs/video/*.webm -c:v libx264 -crf 28 -preset slow
    #             -pix_fmt yuv420p -movflags +faststart docs/demo.mp4
