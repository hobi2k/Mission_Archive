"""미션 18 백엔드용 샘플 영화 데이터를 등록하는 스크립트."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import crud, schemas  # noqa: E402
from app.database import Base, SessionLocal, engine  # noqa: E402


SAMPLE_MOVIES = [
    {
        "title": "기생충",
        "release_date": "2019-05-30",
        "director": "봉준호",
        "genre": "드라마, 스릴러",
        "poster_url": "https://upload.wikimedia.org/wikipedia/en/5/53/Parasite_%282019_film%29.png",
    },
    {
        "title": "올드보이",
        "release_date": "2003-11-21",
        "director": "박찬욱",
        "genre": "스릴러, 미스터리",
        "poster_url": "https://upload.wikimedia.org/wikipedia/en/6/67/Oldboykoreanposter.jpg",
    },
    {
        "title": "인터스텔라",
        "release_date": "2014-11-06",
        "director": "크리스토퍼 놀란",
        "genre": "SF, 드라마",
        "poster_url": "https://upload.wikimedia.org/wikipedia/en/b/bc/Interstellar_film_poster.jpg",
    },
]


def main() -> None:
    """테이블이 비어 있을 때만 샘플 영화를 등록한다."""

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if crud.list_movies(db):
            print("이미 데이터가 존재합니다. 시드 작업을 건너뜁니다.")
            return

        for item in SAMPLE_MOVIES:
            crud.create_movie(db, schemas.MovieCreate(**item))
        print("샘플 영화 3개를 등록했습니다.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
