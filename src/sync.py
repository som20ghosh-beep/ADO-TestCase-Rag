# src/sync.py
import logging
from datetime import datetime

from sqlmodel import select

from .ado_client import ADOClient
from .config import ADO_AREA_PATH, ADO_ORG_URL, ADO_PAT, ADO_PROJECT
from .db import get_session, init_db
from .models import SyncRun, TestCase
from .parser import parse_test_steps

log = logging.getLogger(__name__)


def _parse_ado_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.rstrip("Z"))


def _to_testcase(raw: dict) -> TestCase:
    fields = raw["fields"]
    steps = parse_test_steps(fields.get("Microsoft.VSTS.TCM.Steps"))
    assigned_to = fields.get("System.AssignedTo")
    if isinstance(assigned_to, dict):
        assigned_to = assigned_to.get("displayName")

    return TestCase(
        id=raw["id"],
        title=fields.get("System.Title", ""),
        state=fields.get("System.State", ""),
        priority=fields.get("Microsoft.VSTS.Common.Priority"),
        area_path=fields.get("System.AreaPath", ""),
        iteration_path=fields.get("System.IterationPath"),
        tags=fields.get("System.Tags") or "",
        preconditions=fields.get("Microsoft.VSTS.Common.Description"),
        steps_json=steps,
        expected_result=steps[-1]["expected"] if steps else None,
        automation_status=fields.get("Microsoft.VSTS.TCM.AutomationStatus"),
        assigned_to=assigned_to,
        created_date=_parse_ado_datetime(fields.get("System.CreatedDate")),
        changed_date=_parse_ado_datetime(fields.get("System.ChangedDate")),
        revision=raw.get("rev", 0),
        synced_at=datetime.utcnow(),
        raw_fields=fields,
    )


def get_last_successful_run(session) -> SyncRun | None:
    statement = (
        select(SyncRun)
        .where(SyncRun.error == None)  # noqa: E711
        .where(SyncRun.finished_at != None)  # noqa: E711
        .order_by(SyncRun.finished_at.desc())
    )
    return session.exec(statement).first()


def upsert(session, tc: TestCase) -> None:
    session.merge(tc)


def run_sync(mode: str = "incremental") -> SyncRun:
    init_db()
    session = get_session()
    ado = ADOClient(org_url=ADO_ORG_URL, pat=ADO_PAT, project=ADO_PROJECT)

    run = SyncRun(started_at=datetime.utcnow(), mode=mode)
    session.add(run)
    session.commit()

    try:
        changed_since = None
        if mode == "incremental":
            last_run = get_last_successful_run(session)
            changed_since = last_run.last_changed_date_seen if last_run else None

        ids = ado.query_test_case_ids(ADO_AREA_PATH, changed_since)
        run.items_fetched = len(ids)

        raw_items = ado.get_test_cases_batch(ids) if ids else []
        max_changed = changed_since or datetime.min

        for raw in raw_items:
            try:
                tc = _to_testcase(raw)
                upsert(session, tc)
                run.items_upserted += 1
                if tc.changed_date and tc.changed_date > max_changed:
                    max_changed = tc.changed_date
            except Exception as e:
                run.items_failed += 1
                log.error("failed_item id=%s error=%s", raw.get("id"), e)

        run.last_changed_date_seen = None if max_changed == datetime.min else max_changed
        run.finished_at = datetime.utcnow()
    except Exception as e:
        run.error = str(e)
        run.finished_at = datetime.utcnow()
        raise
    finally:
        session.add(run)
        session.commit()
        session.close()

    return run
