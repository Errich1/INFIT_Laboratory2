from pathlib import Path
from datetime import datetime
import os
import stat

from PIL import Image, ImageFilter

try:
    import numpy as np
except ImportError:
    np = None


# ====== НАСТРОЙКИ ДЛЯ ВАРИАНТА 12 ======
SPATIAL_VARIANT = 1
TONAL_VARIANT = 6

COLOR_STRENGTH = 0.85
SIMULATE_READONLY_AS_DENIED = True

BASE_DIR = Path("Lab2_Media")
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output"
LOG_FILE = BASE_DIR / "logs" / "history.txt"

EXIT_COMMANDS = {"exit", "quit", "q", "выход"}


def parse_color(text: str):
    text = text.strip()
    if not text:
        return None
    if text.startswith("#"):
        text = text[1:]
        if len(text) != 6:
            raise ValueError("HEX-цвет должен быть в формате #RRGGBB")
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    cleaned = text.replace(";", ",").replace(" ", ",")
    parts = [p for p in cleaned.split(",") if p != ""]
    if len(parts) != 3:
        raise ValueError("Нужно три числа: R,G,B")
    values = [int(p) for p in parts]
    for v in values:
        if not (0 <= v <= 255):
            raise ValueError("Каждое значение должно быть в диапазоне 0..255")
    return tuple(values)


def is_read_only(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        st = path.stat()
    except OSError:
        return False
    if os.name == "nt":
        attr = getattr(st, "st_file_attributes", 0)
        return bool(attr & stat.FILE_ATTRIBUTE_READONLY)
    return (st.st_mode & 0o222) == 0


def describe_permissions(path: Path) -> str:
    if not path.exists():
        return "файл не существует"
    try:
        st = path.stat()
    except OSError as exc:
        return f"не удалось прочитать атрибуты: {exc}"
    parts = [
        f"os.access(R_OK)={os.access(path, os.R_OK)}",
        f"os.access(W_OK)={os.access(path, os.W_OK)}",
        f"read_only={is_read_only(path)}",
    ]
    if os.name == "nt":
        attr = getattr(st, "st_file_attributes", 0)
        parts.append(f"FILE_ATTRIBUTE_READONLY={bool(attr & stat.FILE_ATTRIBUTE_READONLY)}")
    else:
        parts.append(f"mode={oct(st.st_mode & 0o777)}")
    return "; ".join(parts)


def apply_spatial(img: Image.Image) -> Image.Image:
    return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=2))


def find_template() -> Path | None:
    for name in ("template.jpg", "template.png"):
        p = INPUT_DIR / name
        if p.exists():
            return p
    return None


def correct_by_template(img: Image.Image) -> Image.Image:
    if np is None:
        print("Предупреждение: numpy не установлен — коррекция по шаблону пропущена.")
        return img
    template_path = find_template()
    if template_path is None:
        print("Предупреждение: шаблон не найден. Положите template.jpg/png в data/input/.")
        return img
    template = Image.open(template_path).convert("RGB").resize(img.size)
    src = np.array(img).astype(np.float32)
    tpl = np.array(template).astype(np.float32)
    src_mean = src.reshape(-1, 3).mean(axis=0)
    src_std = src.reshape(-1, 3).std(axis=0) + 1e-6
    tpl_mean = tpl.reshape(-1, 3).mean(axis=0)
    tpl_std = tpl.reshape(-1, 3).std(axis=0) + 1e-6
    out = (src - src_mean) / src_std * tpl_std + tpl_mean
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def correct_by_color(img: Image.Image, target, strength: float = COLOR_STRENGTH) -> Image.Image:
    if np is None:
        print("Предупреждение: numpy не установлен — коррекция по цвету недоступна.")
        return img
    arr = np.array(img).astype(np.float32)
    mean = arr.reshape(-1, 3).mean(axis=0)
    delta = (np.array(target, dtype=np.float32) - mean) * strength
    return Image.fromarray(np.clip(arr + delta, 0, 255).astype(np.uint8))


def write_log(filename: str, filter_name: str, status: str) -> None:
    try:
        with open(LOG_FILE, "a", encoding="UTF-8") as log:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log.write(f"[{now}] Файл: {filename} | Фильтр: {filter_name} | Статус: {status}\n")
    except Exception as exc:
        print(f"Не удалось записать лог: {exc}")


def process_one_file() -> bool:
    filename = input("\nВведите имя файла (или 'exit' для выхода): ").strip()

    if filename.lower() in EXIT_COMMANDS:
        return False
    if not filename:
        print("Ошибка: имя файла не задано")
        return True

    path = INPUT_DIR / filename
    status = "Ошибка"
    filter_name = "Фильтр Гаусса"

    try:
        if not path.exists():
            raise FileNotFoundError

        # Имитация: image2.png => PermissionError
        if SIMULATE_READONLY_AS_DENIED and filename.lower() == "image2.png":
            if not is_read_only(path):
                print(
                    "Предупреждение: у image2.png нет атрибута 'только чтение', "
                    "но имитация включена. Диагностика:",
                    describe_permissions(path),
                )
            raise PermissionError

        # Реальная проверка права на чтение
        if not os.access(path, os.R_OK):
            raise PermissionError

        with Image.open(path) as im:
            im.load()
            img = im.convert("RGB")

        color_raw = input("Целевой цвет R,G,B или #RRGGBB (Enter — по шаблону): ")
        target = None
        try:
            target = parse_color(color_raw)
        except ValueError as exc:
            print(f"Некорректный цвет ({exc}). Будет использована коррекция по шаблону.")

        img = apply_spatial(img)

        if target is not None:
            img = correct_by_color(img, target)
            filter_name = (
                f"Фильтр Гаусса + коррекция по цвету "
                f"RGB({target[0]},{target[1]},{target[2]})"
            )
        else:
            img = correct_by_template(img)
            filter_name = "Фильтр Гаусса + цветокоррекция по шаблону"

        out_path = OUTPUT_DIR / "result.jpg"
        img.save(out_path, "JPEG", quality=95)
        status = "Успешно"
        print(f"Результат сохранён: {out_path}")

    except FileNotFoundError:
        print("Ошибка: Файл не найден")
        status = "Ошибка: файл не найден"
    except PermissionError:
        print(f"Ошибка: Нет прав на чтение файла {filename}. Обратитесь к администратору")
        status = "Ошибка доступа"
    except Exception as exc:
        print(f"Ошибка обработки: {exc}")
        status = f"Ошибка: {exc}"
    finally:
        write_log(filename, filter_name, status)

    return True


def main():
    print("=== Лабораторная работа №2, вариант 12 ===")
    print("Задание №1: фильтр Гаусса")
    print("Задание №2: цветокоррекция по шаблону (или по введённому цвету)")
    print("Для выхода введите: exit, quit, q или выход (или Ctrl+C)\n")
    try:
        while True:
            if not process_one_file():
                print("Выход из программы.")
                break
    except (KeyboardInterrupt, EOFError):
        print("\nПрервано пользователем. Выход.")


if __name__ == "__main__":
    main()
