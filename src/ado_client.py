# src/ado_client.py
from datetime import datetime

from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication
from tenacity import retry, stop_after_attempt, wait_exponential

class ADOClient:
    def __init__(self, org_url: str, pat: str, project: str):
        creds = BasicAuthentication('', pat)
        self.connection = Connection(base_url=org_url, creds=creds)
        self.wit_client = self.connection.clients.get_work_item_tracking_client()
        self.project = project

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def query_test_case_ids(self, area_path: str, changed_since: datetime | None = None) -> list[int]:
        """Return IDs of test cases matching filters."""
        wiql = f"""
        SELECT [System.Id]
        FROM WorkItems
        WHERE [System.WorkItemType] = 'Test Case'
        AND [System.AreaPath] UNDER '{area_path}'
        """
        if changed_since:
            wiql += f" AND [System.ChangedDate] >= '{changed_since.isoformat()}'"
        result = self.wit_client.query_by_wiql({"query": wiql})
        return [item.id for item in result.work_items]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_test_cases_batch(self, ids: list[int]) -> list[dict]:
        """Fetch full work item details in batches of 200 (ADO limit)."""
        results = []
        for chunk in _chunks(ids, 200):
            items = self.wit_client.get_work_items(ids=chunk, expand="All")
            results.extend([item.as_dict() for item in items])
        return results

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]