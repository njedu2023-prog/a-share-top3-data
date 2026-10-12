import io
import urllib.error
from unittest.mock import patch, MagicMock
import pytest
from orchestrator import top10_chain_orchestrator as m


def test_get_502_recovers_without_dispatch():
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = b'{"status":"in_progress"}'
    response.headers = {'content-type': 'application/json'}
    error = urllib.error.HTTPError(m.API, 502, 'Bad Gateway', {}, io.BytesIO())
    with patch.object(m.urllib.request, 'urlopen', side_effect=[error, response]) as send, patch.object(m.time, 'sleep') as sleep:
        assert m.request(m.API)['status'] == 'in_progress'
        assert send.call_count == 2
        sleep.assert_called_once_with(2)


@pytest.mark.parametrize('method,code,count', [('POST',502,1),('GET',403,1),('GET',502,4)])
def test_unsafe_or_exhausted_requests_fail(method, code, count):
    error = urllib.error.HTTPError(m.API, code, 'error', {}, io.BytesIO())
    with patch.object(m.urllib.request, 'urlopen', side_effect=error) as send, patch.object(m.time, 'sleep'):
        with pytest.raises(urllib.error.HTTPError):
            m.request(m.API, method=method)
        assert send.call_count == count


@pytest.mark.parametrize('csv,valid', [('trade_date,run_id\n20261012,42\n',True),('trade_date,run_id\n20261009,42\n',False),('trade_date,run_id\n20261012,41\n',False),('trade_date,run_id\n',False)])
def test_top10_reuse_requires_exact_date_and_success_identity(csv, valid):
    run = {'id':42, 'html_url':'https://example.invalid/42'}
    with patch.object(m, 'recent_success', return_value=run), patch.object(m, 'get_text', return_value=csv), patch.object(m, 'dispatch') as dispatch:
        if valid:
            assert m.completed_daily_stage(m.TOP10_REPO,m.TOP10_WF,'token','20261012') == run
        else:
            with pytest.raises(m.ChainError, match='refusing duplicate'):
                m.completed_daily_stage(m.TOP10_REPO,m.TOP10_WF,'token','20261012')
        dispatch.assert_not_called()


def test_reuse_decision_does_not_depend_on_premium():
    run = {'id':43,'html_url':'https://example.invalid/43'}
    with patch.object(m,'recent_success',return_value=run), patch.object(m,'decision_dashboard_data_status',return_value=(True,'')), patch.object(m,'decision_report_for_signal',return_value='report'), patch.object(m,'chain_outputs_status',side_effect=AssertionError('Premium must not gate stage reuse')):
        assert m.completed_daily_stage(m.DECISION_REPO,m.DECISION_WF,'token','20261012') == run


def test_absent_success_allows_normal_execution():
    with patch.object(m,'recent_success',return_value=None):
        assert m.completed_daily_stage(m.TOP10_REPO,m.TOP10_WF,'token','20261012') is None
