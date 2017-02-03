import arim
import pytest

class TestConfig:
    @pytest.fixture()
    def conf(self):
        return arim.config.Config([
            ('numelements', 1),
            ('name', 'Foo Bar'),
            ('pi_list' , [3, 1, 4, 1, 6]),

            ('submodule', dict(subval1=1, subval2=2))
        ])

    def test_config(self, conf):
        conf['bar'] = 2
        conf['bar.baz'] = 3

        str(conf)
        repr(conf)

        assert conf['numelements'] == 1
        assert conf['name'] == 'Foo Bar'

        assert conf.find_all('bar') == arim.config.Config([('bar', 2), ('bar.baz', 3)])
        assert conf.keys() == sorted(conf.keys())


    def test_config_merge(self, conf):
        conf2 = dict(numelements=2, newval=666, submodule=dict(subval1=777, newval9=9))

        assert conf['numelements'] == 1
        assert conf['name'] == 'Foo Bar'
        assert conf['submodule']['subval1'] == 1
        with pytest.raises(KeyError):
            conf['submodule']['newval9']

        # merge:
        conf.merge(conf2)
        assert type(conf) is arim.config.Config

        assert conf['numelements'] == 2
        assert conf['name'] == 'Foo Bar'
        assert conf['submodule']['subval1'] == 777
        assert conf['submodule']['newval9'] == 9

