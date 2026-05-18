import csv
from hypdelay.core.config import Config

def save_all_results(results, filename='results.csv'):
    """
    Сохраняет список результатов в CSV-файл.

    Параметры:
        results : list
            Список элементов, каждый из которых может быть:
                - объектом Config (тогда будут сохранены все его параметры)
                - кортежем (cfg, extra_dict), где cfg — Config, а extra_dict — словарь дополнительных полей
                - обычным словарём (прямое сохранение)
        filename : str
            Имя выходного CSV-файла.
    """
    rows = []
    for item in results:
        if isinstance(item, Config):
            row = _config_to_row(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], Config):
            cfg, extra = item
            row = _config_to_row(cfg)
            row.update(extra)
        elif isinstance(item, dict):
            row = item
        else:
            raise TypeError(f"Неподдерживаемый тип элемента: {type(item)}")
        rows.append(row)

    if not rows:
        return

    # Определяем все возможные ключи (объединение всех словарей)
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    # Сортируем для удобочитаемости (но можно оставить и без сортировки)
    fieldnames = sorted(all_keys)

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Сохранено {len(rows)} записей в {filename}")


def _config_to_row(cfg: Config):
    """Внутренняя функция: преобразует объект Config в словарь для CSV."""
    model = cfg.model
    model_cls = model.__class__

    # Параметры модели (из param_info)
    model_params = {}
    for name, default, _ in model_cls.param_info():
        # Берём значение из экземпляра модели, если есть, иначе default
        value = getattr(model, name, default)
        model_params[name] = value

    # Параметры схемы и сетки (из cfg)
    scheme_params = {
        'x0': cfg.x0,
        't0': cfg.t0,
        's': cfg.s,
        'L': cfg.L,
        'T': cfg.T,
        'N': cfg.N,
        'M': cfg.M,
        'h': cfg.h,
        'delta': cfg.delta,
    }

    # Объединяем с именем модели
    row = {'model': model_cls.__name__}
    row.update(model_params)
    row.update(scheme_params)

    # Добавляем производные отношения, если их нет
    if cfg.N != 0:
        row['M/N'] = cfg.M / cfg.N
    if cfg.delta != 0:
        row['h/delta'] = cfg.h / cfg.delta

    return row